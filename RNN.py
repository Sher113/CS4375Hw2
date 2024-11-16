import torch
import torch.nn as nn
import torch.nn.functional as F

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, drop_prob=0.5):
        super(SentimentRNN, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layer
        self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, nonlinearity='tanh', batch_first=True)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # Linear layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # LogSoftmax activation
        self.log_softmax = nn.LogSoftmax(dim=1)
        
        # Loss function
        self.loss = nn.NLLLoss()
        
    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, x):
        # embeddings and RNN output
        embeds = self.embedding(x)
        rnn_out, hidden = self.rnn(embeds)
        
        # Take the output of the last time step
        rnn_out = rnn_out[:, -1, :]
        
        # Pass through dropout layer
        out = self.dropout(rnn_out)
        
        # Fully connected layer
        out = self.fc(out)
        
        # LogSoftmax activation
        predicted_vector = self.log_softmax(out)
        
        return predicted_vector

# Hyperparameters (example values)
vocab_size = 5000  # Size of the vocabulary
embedding_dim = 300  # Dimension of word embeddings
hidden_dim = 128  # Hidden layer dimension
output_dim = 5  # Number of classes (for 5-class sentiment analysis)
n_layers = 1  # Number of RNN layers

# Model instantiation
model = SentimentRNN(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers)

# Example input (batch of sequences)
x_example = torch.randint(0, vocab_size, (32, 100))  # Batch size 32, sequence length 100

# Forward pass
output = model(x_example)
print(output)
