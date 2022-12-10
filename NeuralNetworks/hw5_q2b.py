import math
import numpy as np
import pandas as pd 
from network import *

def cost(y, x, NN):
    jw = 0
    for i in range(len(y)):
        y_out = np.sign(NN.forward(x[i]))
        if y[i]!=y_out:
            jw += 1
    return jw/len(y)


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

if __name__ == "__main__":
    ## load the data
    x_test, y_test = load_data("bank-note/test.csv")
    train_data = load_data_cmp("bank-note/train.csv")
    train_data[:,-1][train_data[:,-1]==0] = -1
    y_test[y_test==0] = -1
    np.random.seed(200)

    
    #print(x_test.shape[1])
    width = [5,10,25,50,100]
    for w in width:
        r_init = 0.01
        nn = NN(nh1=w, nh2=w, inp=x_test.shape[1], wt_type="randn")
        print(f'Training for width of inner layer={w}')
        for epoch in range(20):
            train_data_new = train_data.copy()
            np.random.shuffle(train_data_new)
            x_train, y_train = train_data_new[:,:-1], train_data_new[:,-1]
            r = r_init/(1+((r_init/0.1)*epoch))
            for i in range(len(y_train)):
                y_out = nn.forward(x_train[i])
                dw = nn.backward(y=y_out, ys=y_train[i])
                nn.update_weights(dw, r)
            loss = cost(y_train, x_train, nn)
            print(f"Error at t={epoch} is {loss}")

        test_cost = cost(y_test, x_test, nn)
        print(f"Errors for width:{w} : Train={loss}, Test={test_cost}")
        