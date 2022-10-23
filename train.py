# pytorch cnn for multiclass classification
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from model import CNN
from data import prepare_data

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# train the model


def train_model(train_dl, model):
    # define the optimization
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    # enumerate epochs
    for epoch in range(10):
        # enumerate mini batches
        for i, (inputs, targets) in tqdm(enumerate(train_dl)):
            inputs = inputs.to(device)
            targets = targets.to(device)

        # clear the gradients

        optimizer.zero_grad()
        # compute the model output
        yhat = model(inputs)
        # calculate loss
        loss = criterion(yhat, targets)
        # credit assignment
        loss.backward()
        # update model weights
        optimizer.step()


# prepare the data
path = '~/.torch/datasets/mnist'
train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))

# define the network
model = CNN(1).to(device)
# # train the model
train_model(train_dl, model)

PATH = '/home/saadmann/repos/Multiclass-NN-terminal-setup/model'

torch.save(model, PATH)

##########

# evaluate the model
