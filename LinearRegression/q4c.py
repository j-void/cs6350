import math
import numpy as np
import pandas as pd 

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
    
    X = x_train.T
    Y = y_train.T
    
    A = X.dot(X.T)
    B = X.dot(Y)
    W = np.linalg.inv(A).dot(B)
    print("Optimal weight vector =",W)
    

        