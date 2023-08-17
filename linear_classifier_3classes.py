import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sklearn import metrics
from sklearn import decomposition
from sklearn import manifold
from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt
import numpy as np

import copy
import random
import time
import classifier_load_dataset
import itertools
import json



class MLP(nn.Module):
    def __init__(self, input_dim, output_dims):
        super().__init__()

        self.output_fc0 = nn.Linear(input_dim, output_dims[0])
        self.output_fc1 = nn.Linear(input_dim, output_dims[1])
        self.output_fc2 = nn.Linear(input_dim, output_dims[2])

    def forward(self, x):
        batch_size = x.shape[0]
        x = x[:,:3,:,:].reshape(batch_size, -1)

        y_pred = {}
        y_pred[0] = self.output_fc0(x)
        y_pred[1] = self.output_fc1(x)
        y_pred[2] = self.output_fc2(x)

        return y_pred 



def train(model, iterator, optimizer, criterion, device):

    epoch_loss = 0
    epoch_acc = {0: 0, 1: 0, 2: 0}
    model.train()

    for (x, y) in tqdm(iterator, desc="Training", leave=False):
        x = x.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred[0], y[0]) + criterion(y_pred[1], y[1]) + criterion(y_pred[2], y[2])
        acc = {}
        acc[0] = calculate_accuracy(y_pred[0], y[0])
        acc[1] = calculate_accuracy(y_pred[1], y[1])
        acc[2] = calculate_accuracy(y_pred[2], y[2])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc[0] += acc[0].item()
        epoch_acc[1] += acc[1].item()
        epoch_acc[2] += acc[2].item()

    epoch_acc[0] /= len(iterator)
    epoch_acc[1] /= len(iterator)
    epoch_acc[2] /= len(iterator)

    return epoch_loss / len(iterator), epoch_acc 

def calc_mse(pred, gt): 
    return torch.sqrt(torch.mean((pred-gt)**2))
    

def evaluate(model, iterator, criterion, device):

    epoch_loss = 0
    #epoch_acc = 0
    epoch_acc = {0: 0, 1: 0, 2: 0}

    model.eval()
    with torch.no_grad():
        for (x, y) in tqdm(iterator, desc="Evaluating", leave=False):
            x = x.to(device)
            #y = _y[key].to(device)

            y_pred = model(x)
            loss = criterion(y_pred[0], y[0]) + criterion(y_pred[1], y[1]) + criterion(y_pred[2], y[2])
            acc = {}
            acc[0] = calculate_accuracy(y_pred[0], y[0])
            acc[1] = calculate_accuracy(y_pred[1], y[1])
            acc[2] = calculate_accuracy(y_pred[2], y[2])
            epoch_loss += loss.item()
            #epoch_acc += acc.item()
            epoch_acc[0] += acc[0].item()
            epoch_acc[1] += acc[1].item()
            epoch_acc[2] += acc[2].item()

    epoch_acc[0] /= len(iterator)
    epoch_acc[1] /= len(iterator)
    epoch_acc[2] /= len(iterator)

    return epoch_loss / len(iterator), epoch_acc #/ len(iterator)


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == "__main__": 

    dataset = "single-body_2d_3classes"
    properties_json = "properties_"+dataset+".json"
    with open(properties_json, 'r') as f: 
        properties = json.load(f)

    keys, values = zip(*properties.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    configs = []
    for permutation in permutations:
        configs.append("".join(permutation.values()))
    
    
    pixel_size = 28
    n_class_color = 2 #3

    tf = transforms.Compose([transforms.Resize((pixel_size,pixel_size)), transforms.ToTensor()])
    train_dataset = classifier_load_dataset.my_dataset(tf, 5000, dataset, configs=configs, training=True, n_class_color=n_class_color)
    test_data = classifier_load_dataset.my_dataset(tf, 500, dataset, configs=configs, training=True, n_class_color=n_class_color)

    train_data, valid_data = data.random_split(train_dataset, [4500, 500])
    BATCH_SIZE = 128
    valid_data = copy.deepcopy(valid_data)
    train_iterator = data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    valid_iterator = data.DataLoader(valid_data, batch_size=BATCH_SIZE)
    test_iterator = data.DataLoader(test_data, batch_size=BATCH_SIZE)
    
    
    INPUT_DIM = pixel_size * pixel_size * 3 #4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    #for key in ["shapes","colors","sizes"]:
    OUTPUT_DIMS = [len(properties[key]) for key in ["shapes","colors","sizes"]]
    OUTPUT_DIMS[1] = n_class_color
    model = MLP(INPUT_DIM, OUTPUT_DIMS)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    #model = model.to(device)
    criterion = criterion.to(device)
    
    EPOCHS = 10 #200
    best_valid_loss = float('inf')
    
    for epoch in trange(EPOCHS):
    
        start_time = time.monotonic()
    
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)
    
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'working/linear-classifier_'+dataset+'_multi-class.pt')
    
        end_time = time.monotonic()
    
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc[0]*100:.2f}% {train_acc[1]*100:.2f}% {train_acc[2]*100:.2f}%')
        print(f'\tValid Loss: {valid_loss:.3f} | valid Acc: {valid_acc[0]*100:.2f}% {valid_acc[1]*100:.2f}% {valid_acc[2]*100:.2f}%')
    
    
    test_loss, test_acc = evaluate(model, test_iterator, criterion, device)
    #print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc[0]*100:.2f}%')
    print(f'\tTest Loss: {test_loss:.3f} | test Acc: {test_acc[0]*100:.2f}% {test_acc[1]*100:.2f}% {test_acc[2]*100:.2f}%')


