import math
import numpy as np
import pandas as pd 

def cost(y, x, w):
    jw = 0
    for i in range(len(y)):
        x_ = np.append(x[i], 1)
        y_out = np.sign(np.dot(w.transpose(),x_))
        if y[i]!=y_out:
            jw += 1
    return jw/len(y)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

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
    np.random.seed(2)
    ## intialize the weight vector
    w_init = np.zeros((train_data[:,:-1].shape[1]+1), dtype=float)
    
    r_init = 0.01
    variance = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
    for v in variance:
        w = w_init
        for epoch in range(100):
            train_data_new = train_data.copy()
            #np.random.shuffle(train_data_new)
            x_train, y_train = train_data_new[:,:-1], train_data_new[:,-1]
            r = r_init/(1+((r_init/0.1)*epoch))
            m = len(y_train)
            for i in range(len(y_train)):
                x_ = np.append(x_train[i], 1)
                #print((1-sigmoid(-np.dot(w.transpose(),x_)*y_train[i])))
                grad =  m * (sigmoid(y_train[i]*np.dot(w.transpose(), x_))-1) * (y_train[i]*x_)
                w = w - r*grad
            loss = cost(y_train, x_train, w)
            #print(f"Error at t={epoch} is {loss}, w={w}")
    
        test_cost = cost(y_test, x_test, w)
        print(f"Errors for variance:{v} : Train={loss}, Test={test_cost}")
        