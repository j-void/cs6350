import math
import numpy as np
import pandas as pd 


def cost(y, x, w):
    jw = 0
    for i in range(len(y)):
        x_ = np.append(x[i], 1)
        y_out = np.sign(np.dot(w.transpose(),x_)) #np.sign(np.dot(w.transpose(),x[i]))
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
    
    ## intialize the weight vector
    w_init = np.zeros((train_data[:,:-1].shape[1]+1), dtype=float)
    #12
    np.random.seed(12)
    C_list = [100/873, 500/873, 700/873]
    
    for C in C_list:
        r_init = 1e-3
        w = w_init
        for epoch in range(100):
            train_data_new = train_data.copy()
            np.random.shuffle(train_data_new)
            x_train, y_train = train_data_new[:,:-1], train_data_new[:,-1]
            r = r_init/(1+epoch)
            for i in range(len(y_train)):
                x_ = np.append(x_train[i], 1)
                if y_train[i]*np.dot(w.transpose(),x_)<=1:
                    w = w - r*np.append(w[:-1],0) + r*C*y_train.shape[0]*y_train[i]*x_
                else:
                    w[:-1] = w[:-1] - r*w[:-1]
        train_cost = cost(y_train, x_train, w)
        print(f"Final weights={w} for C={C} & final lr={r}")
        test_cost = cost(y_test, x_test, w)
        print(f"Error for C={C} : Train={train_cost}; Test={test_cost}")
    
    
        