import math
import numpy as np
import pandas as pd 


def cost(y, x, w):
    jw = 0
    for i in range(len(y)):
        y_out = np.sign(np.dot(w.transpose(),x[i]))
        if y[i]!=y_out:
            jw += 1
    return jw/len(y)

def cost_c(y, x, c, g):
    jw = 0
    for i in range(len(y)):
        dist = np.linalg.norm(x-x[i], axis=1)
        kernel = np.exp(-(dist**2/g))
        y_out = np.sum(np.einsum('i,i,i->i', c, y, kernel))
        #print(y[i], y_out)
        if y[i]!=np.sign(y_out):
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
    np.random.seed(1)


    w_init = np.zeros((train_data[:,:-1].shape[1]), dtype=float)
    
    gammas = [0.1, 0.5, 1, 5, 100]
    for g in gammas:
        w = w_init
        c = np.zeros((len(train_data)), dtype=float)
        for epoch in range(100):
            train_data_new = train_data.copy()
            np.random.shuffle(train_data_new)
            x_train, y_train = train_data_new[:,:-1], train_data_new[:,-1]
            for i in range(len(y_train)):
                dist = np.linalg.norm(x_train-x_train[i], axis=1)
                kernel = np.exp(-(dist**2/g))
                y_out = np.sign(np.sum(np.einsum('i,i,i->i', c, y_train, kernel)))
                if y_train[i]!=y_out:
                    c[i] = c[i] + 1
        w = np.einsum('i,i,ik->k', c, y_train, x_train)
        #train_cost = cost_c(y_train, x_train, c, g)
        train_cost = cost(y_train, x_train, w)
        test_cost = cost(y_test, x_test, w)
        print(f"Final weights={w} for gamma={g}")
        print(f"Error for gamma={g} : Train={train_cost}; Test={test_cost}")
        