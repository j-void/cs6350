import math
import numpy as np
import pandas as pd 


def cost(y, x, weights, confidence):
    jw = 0
    for i in range(len(y)):
        y_out = 0
        for j in range(len(weights)):
            y_out +=  confidence[j]*np.sign(np.dot(weights[j].transpose(),x[i]))
        y_out = np.sign(y_out)
        if y[i] != y_out:
            jw += 1
    return jw/len(y)

def load_data_cmp(path):
    data = []
    with open(path , 'r') as f : 
        for line in f :
            terms = line.strip().split(',')
            data.append(terms)
    return np.array(data).astype(float)

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
    x_test, y_test = load_data("bank-note/test.csv")
    train_data = load_data_cmp("bank-note/train.csv")
    train_data[:,-1][train_data[:,-1]==0] = -1
    y_test[y_test==0] = -1
    
    ## intialize the weight vector
    w_init = np.zeros((train_data[:,:-1].shape[1]), dtype=float)
    r = 0.001
    steps = []
    costs = []
    w = w_init
    cm = []
    m = 0
    weights = []
    cm_var = 1
    np.random.seed(20)
    for epoch in range(10):
        train_data_new = train_data.copy()
        np.random.shuffle(train_data_new)
        x_train, y_train = train_data_new[:,:-1], train_data_new[:,-1]
        for i in range(len(y_train)):
            y_out = np.sign(np.dot(w.transpose(),x_train[i]))
            if y_train[i] != y_out:
                weights.append(w)
                cm.append(cm_var)
                w = w + r*y_train[i]*x_train[i]
                m += 1
                cm_var = 1
            else:
                if not weights:
                    weights.append(w)
                    cm.append(1)
                cm_var += 1
                
                

    for i in range(len(weights)):
        print(f"weights: {weights[i]}, confidence: {cm[i]}")

    test_cost = cost(y_test, x_test, weights, cm)
    print(f"Test Error={test_cost}")
    
    # import pickle
    # error_dict = {"costs":costs, "test_cost":test_cost, "lr":r, "w":w}
    # with open('q4b_out.pkl', 'wb') as f:
    #     pickle.dump(error_dict, f)
    
        