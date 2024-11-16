# FFNN Implementation for CS4375 Homework 2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_loader import load_data  # Assuming a load_data function is provided for loading datasets

class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        z = self.fc2(h)
        return z

# Hyperparameters and Model Training for FFNN
input_dim = 5000  # Example vocabulary size
hidden_dim = 128
output_dim = 5  # Output classes (ratings from 1 to 5)
learning_rate = 0.001
epochs = 10
batch_size = 64

train_data, val_data, _ = load_data("path/to/train_data", "path/to/val_data")
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

model = FeedForwardNN(input_dim, hidden_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

def train_ffnn(model, train_loader, val_loader, optimizer, criterion, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {total_loss/len(train_loader):.4f}")

        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        val_accuracy = 100 * correct / total
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_accuracy:.2f}%")

train_ffnn(model, train_loader, val_loader, optimizer, criterion, epochs)


# RNN Implementation for CS4375 Homework 2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_loader import load_data  # Assuming a load_data function is provided for loading datasets

class RecurrentNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(RecurrentNN, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        h, _ = self.rnn(x)
        h = h[:, -1, :]  # Take the last hidden state
        z = self.fc(h)
        return z

# Hyperparameters and Model Training for RNN
input_dim = 5000  # Example vocabulary size
embedding_dim = 100
hidden_dim = 128
output_dim = 5  # Output classes (ratings from 1 to 5)
learning_rate = 0.001
epochs = 10
batch_size = 64

train_data, val_data, _ = load_data("path/to/train_data", "path/to/val_data")
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

model = RecurrentNN(input_dim, embedding_dim, hidden_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

def train_rnn(model, train_loader, val_loader, optimizer, criterion, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {total_loss/len(train_loader):.4f}")

        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        val_accuracy = 100 * correct / total
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_accuracy:.2f}%")

train_rnn(model, train_loader, val_loader, optimizer, criterion, epochs)
