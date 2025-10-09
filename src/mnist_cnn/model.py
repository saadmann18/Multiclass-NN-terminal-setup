"""
CNN model definition for MNIST digit classification.
"""

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

    def __init__(self, n_channels: int = 1):
        super(CNN, self).__init__()

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

        # Output layer
        self.hidden4 = Linear(100, 10)
        xavier_uniform_(self.hidden4.weight)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, 10)
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
        x = x.view(x.size(0), 5 * 5 * 32)

        # Fully connected layers
        x = self.hidden3(x)
        x = self.act3(x)

        # Output layer
        x = self.hidden4(x)
        return x
