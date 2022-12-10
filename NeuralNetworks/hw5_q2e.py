import math
import numpy as np
import pandas as pd 
import torch
from tqdm import tqdm
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader

def cost(data_loader, model):
    jw = 0
    for x, y in data_loader:
        y_out = 1 if model(x).squeeze() > 0.5 else 0
        if y!=y_out:
            jw += 1
    return jw/len(data_loader)


def load_data(path):
    y = []
    x = []
    with open(path , 'r') as f : 
        for line in f :
            terms = line.strip().split(',')
            x.append(terms[:-1])
            y.append(terms[-1])
    return np.array(x).astype(float), np.array(y).astype(float)

def load_data_cmp(path):
    data = []
    with open(path , 'r') as f : 
        for line in f :
            terms = line.strip().split(',')
            data.append(terms)
    return np.array(data).astype(float)

class TDataset(Dataset):
    def __init__(self, path):
        self.x, self.y = load_data(path)
    
    def input_dim(self):
        return self.x.shape[1]
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]).float(),  torch.tensor([self.y[idx]]).float()
    
def weight_init(module, initf):
    def foo(m):
        classname = m.__class__.__name__.lower()
        if isinstance(m, module):
            initf(m.weight)
    return foo 

class NeuralNtwrk(nn.Module):
    def __init__(self, depth, width, ilen, activation='tanh'):
        super().__init__()
        self.layers = nn.ModuleList()
        if activation=='RELU':
            self.activation_fn = nn.ReLU()
            self.initfn = nn.init.kaiming_uniform_
            print("Using 'he' initialization.")
        else:
            self.activation_fn = nn.Tanh()
            self.initfn = nn.init.xavier_normal_
            print("Using 'xavier' initialization.")
        
        layer_zero = nn.Sequential(
                nn.Linear(ilen, width),
                self.activation_fn,
            )
        self.layers.append(layer_zero)
        
        for i in range(depth):
            layer = nn.Sequential(
                nn.Linear(width, width),
                self.activation_fn,
            )
            self.layers.append(layer)
        
        ## Final layer
        self.layers.append(nn.Linear(width, 1))
        
        self.apply(weight_init(module=nn.Linear, initf=self.initfn))
        
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return torch.sigmoid(x)
    


if __name__ == "__main__":
    ## load the data
    train_data = TDataset("bank-note/train.csv")
    test_data = TDataset("bank-note/test.csv")
    
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
    
    depth = [3, 5, 9]
    widths = [5, 10, 25, 50, 100]
    activations = ['tanh', 'RELU'] 
    
    for afn in activations:
        for d in depth:
            for w in widths:
                tqdm.write('Training for depth='+str(d)+', width='+str(w)+' and activation='+str(afn))
                lr = 1e-3
                model = NeuralNtwrk(depth=d, width=w, ilen=train_data.input_dim(), activation=afn)
                criterion = nn.MSELoss()
                optim = torch.optim.Adam(model.parameters(), lr, betas=(0.9, 0.999))
                optim.zero_grad()
                
                for epoch in tqdm(range(10), desc="Epochs: "):
                    loss_list = []
                    for x, y in train_loader:
                        optim.zero_grad()
                        inp_ = torch.tensor
                        y_out = model(x)
                        loss  = criterion(y_out, y)
                        loss.backward()
                        optim.step()
                        loss_list.append(loss.detach().squeeze())
                    tqdm.write('Epoch - '+str(epoch)+": MSE="+str(np.mean(np.array(loss_list))))  
                train_err = cost(train_loader, model)
                test_err = cost(test_loader, model)
                tqdm.write('Errors for depth='+str(d)+', width='+str(w)+' and activation='+str(afn)+': Train='+str(train_err)+', Test='+str(test_err))  
        