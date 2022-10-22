import math
import numpy as np
import pandas as pd 

def cost(y, w, x):
    jw = 0
    for i in range(len(y)):
        jw += (y[i]-np.dot(w.transpose(),x[i]))**2
    return jw/2

def load_data(path):
    y = []
    x = []
    with open(path , 'r') as f : 
        for line in f :
            terms = line.strip().split(',')
            x.append(terms[:-1])
            y.append(terms[-1])
    return np.array(x).astype(float), np.array(y).astype(float)

if __name__ == "__main__":
    ## load the data
    x_train, y_train = load_data("concrete/train.csv")
    
    # ones = np.ones([x_train.shape[0],x_train.shape[1]+1])
    # ones[:,1:] = x_train
    X = x_train.T#ones.T
    #print(X.shape)
    Y = y_train.T
    
    A = X.dot(X.T)
    B = X.dot(Y)
    W = np.linalg.inv(A).dot(B)
    print("Optimal weight vector =",W)
    
    train_cost = cost(y_train, W, x_train)
    print("Train cost for optimal W =",train_cost)
    

        