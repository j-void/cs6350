import math
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

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
    return np.array(x).astype(np.float), np.array(y).astype(np.float)

if __name__ == "__main__":
    ## load the data
    x_train, y_train = load_data("concrete/train.csv")
    x_test, y_test = load_data("concrete/test.csv")
    
    ## intialize the weight vector
    w = np.zeros((x_train.shape[1]), dtype=float)
    r = 0.001
    steps = []
    costs = []
    
    for epoch in range(500):
        diff = 0
        for j in range(len(w)):
            dw = gradient(y_train, w, x_train, j)
            w[j] = w[j] - r*dw
        loss = cost(y_train, w, x_train)
        print(f"Loss at t={epoch} = {loss}, w={w}, lr={r}")
        steps.append(epoch)
        costs.append(loss)

    
    print(f"Final lr={r} and weights={w}")
    test_cost = cost(y_test, w, x_test)
    print(f"Test Cost={test_cost}")
    
    import pickle
    error_dict = {"costs":costs, "test_cost":test_cost, "lr":r, "w":w}
    with open('q4a_out.pkl', 'wb') as f:
        pickle.dump(error_dict, f)
    
    fig, ax = plt.subplots()
    ax.plot(np.array(steps), np.array(costs), '-bo')
    ax.set(xlabel='steps', ylabel='cost')
    fig.savefig("q4a.png")
    #plt.show()
        