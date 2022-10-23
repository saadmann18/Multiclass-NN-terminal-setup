from numpy import vstack
from numpy import argmax
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
from model import CNN
from data import prepare_data

device = "cuda" if torch.cuda.is_available() else "cpu"
 
# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in tqdm(enumerate(test_dl)):
        inputs = inputs.to(device)
        targets = targets.to(device)
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.cpu().detach().numpy()
        actual = targets.cpu().numpy()
        # convert to class labels
        yhat = argmax(yhat, axis=1)
        # reshape for stacking
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.reshape((len(yhat), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc

# prepare the data
path = '~/.torch/datasets/mnist'
train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))

# define the network
PATH = '/home/saadmann/repos/Multiclass-NN-terminal-setup/model'
model = torch.load(PATH)
model.eval()

# evaluate the model
acc = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc)
