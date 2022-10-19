import math
import numpy as np
from utils import *
import pandas as pd 


def cost(y, w, x):
    jw = 0
    for i in range(len(y)):
        jw += (y[i]-np.dot(w.transpose(),x[i]))**2
    return jw/2

def gradient(y, w, x, j):
    a = 0
    for i in range(len(y)):
        a += (y[i]-np.dot(w.transpose(),x[i]))*x[i,j]
    return -a

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
    x_test, y_test = load_data("concrete/test.csv")
    
    ## intialize the weight vector
    w_init = np.zeros((x_train.shape[1]))
    r = 0.03125
    steps = []
    costs = []
    w = w_init
    for epoch in range(100):
        diff = 0
        for i in range(len(y_train)):
            w = w + r*(y_train[i]-np.dot(w.transpose(),x_train[i]))*x_train[i]
        loss = cost(y_train, w, x_train)
        print(f"Loss at t={epoch} is {loss}, w={w}")
        steps.append(epoch)
        costs.append(loss)
        if loss < 1e-1:
            break

    
    print(f"Final lr={r} and weights={w}")
    test_cost = cost(y_test, w, x_test)
    print(f"Test Cost={test_cost}")
    
    import pickle
    error_dict = {"costs":costs, "test_cost":test_cost, "lr":r, "w":w}
    with open('q4b_out.pkl', 'wb') as f:
        pickle.dump(error_dict, f)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(np.array(steps), np.array(costs), '-bo')
    ax.set(xlabel='steps', ylabel='cost')
    fig.savefig("q4b.png")
    #plt.show()
        