"""
CNN model definition for MNIST digit classification.
"""

import torch
from torch.nn import Conv2d, MaxPool2d, Linear, ReLU, Module
from torch.nn.init import kaiming_uniform_, xavier_uniform_


class CNN(Module):
    """
    Convolutional Neural Network for MNIST digit classification.

    Architecture:
    - Conv2d(1, 32, 3x3) -> ReLU -> MaxPool2d(2x2)
    - Conv2d(32, 32, 3x3) -> ReLU -> MaxPool2d(2x2)
    - Linear(5*5*32, 100) -> ReLU
    - Linear(100, 10) -> Output

    Args:
        n_channels (int): Number of input channels (1 for grayscale MNIST)
    """

    def __init__(self, n_channels: int = 1, embedding_dim: int = 2):
        super(CNN, self).__init__()
        self.embedding_dim = embedding_dim

        # First convolutional block
        self.hidden1 = Conv2d(n_channels, 32, (3, 3))
        kaiming_uniform_(self.hidden1.weight, nonlinearity="relu")
        self.act1 = ReLU()
        self.pool1 = MaxPool2d((2, 2), stride=(2, 2))

        # Second convolutional block
        self.hidden2 = Conv2d(32, 32, (3, 3))
        kaiming_uniform_(self.hidden2.weight, nonlinearity="relu")
        self.act2 = ReLU()
        self.pool2 = MaxPool2d((2, 2), stride=(2, 2))

        # Fully connected layers
        self.hidden3 = Linear(5 * 5 * 32, 100)
        kaiming_uniform_(self.hidden3.weight, nonlinearity="relu")
        self.act3 = ReLU()

        # Embedding layer
        self.embedding = Linear(100, embedding_dim)
        kaiming_uniform_(self.embedding.weight, nonlinearity='linear')

        # Output layer
        self.hidden4 = Linear(embedding_dim, 10)
        xavier_uniform_(self.hidden4.weight)

    def forward(self, x, return_embeddings: bool = False):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
            return_embeddings (bool): If True, returns both logits and embeddings

        Returns:
            torch.Tensor: Output logits of shape (batch_size, 10)
            or tuple: (logits, embeddings) if return_embeddings is True
        """
        # First convolutional block
        x = self.hidden1(x)
        x = self.act1(x)
        x = self.pool1(x)

        # Second convolutional block
        x = self.hidden2(x)
        x = self.act2(x)
        x = self.pool2(x)

        # Flatten for fully connected layers
        x = x.reshape(x.size(0), -1)  # Using reshape instead of view for better compatibility

        # Fully connected layers
        x = self.hidden3(x)
        x = self.act3(x)
        
        # Get embeddings
        embeddings = self.embedding(x)
        
        # Output layer
        logits = self.hidden4(embeddings)
        
        if return_embeddings:
            return logits, embeddings
        return logits
        
    def get_embeddings(self, x):
        """
        Get embeddings for input x.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            torch.Tensor: Embeddings of shape (batch_size, embedding_dim)
        """
        with torch.no_grad():
            # First convolutional block
            x = self.hidden1(x)
            x = self.act1(x)
            x = self.pool1(x)
            
            # Second convolutional block
            x = self.hidden2(x)
            x = self.act2(x)
            x = self.pool2(x)
            
            # Flatten and get embeddings
            x = x.reshape(x.size(0), -1)
            x = self.hidden3(x)
            x = self.act3(x)
            embeddings = self.embedding(x)
            
        return embeddings
