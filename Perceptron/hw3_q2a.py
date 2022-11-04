import math
import numpy as np
import pandas as pd 


def cost(y, w, x):
    jw = 0
    for i in range(len(y)):
        y_out = np.sign(np.dot(w.transpose(),x[i]))
        if y[i]*y_out<=0:
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
    ## intialize the weight vector
    w_init = np.zeros((train_data[:,:-1].shape[1]), dtype=float)
    r = 0.1
    w = w_init
    for epoch in range(10):
        train_data_new = train_data.copy()
        np.random.shuffle(train_data_new)
        #print(train_data[0])
        x_train, y_train = train_data_new[:,:-1], train_data_new[:,-1]
        #print(y_train)
        for i in range(len(y_train)):
            y_out = np.sign(np.dot(w.transpose(),x_train[i]))
            if y_train[i]*y_out<=0:
                w = w + r*y_train[i]*x_train[i]
        # loss = cost(y_train, w, x_train)
        # print(f"Error at t={epoch} is {loss}, w={w}")


    
    print(f"Final lr={r} and weights={w}")
    test_cost = cost(y_test, w, x_test)
    print(f"Test Error={test_cost}")
        