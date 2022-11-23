import math
import numpy as np
import pandas as pd 
from scipy import optimize

def cost(y, x, w):
    jw = 0
    for i in range(len(y)):
        x_ = np.append(x[i], 1)
        y_out = np.sign(np.dot(w.transpose(),x_))
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
    w_init = np.zeros((train_data[:,:-1].shape[1]), dtype=float)
    C_list = [100/873, 500/873, 700/873]
    train_data_new = train_data.copy()
    x_train, y_train = train_data_new[:,:-1], train_data_new[:,-1]
    
    for C in C_list:
        w = w_init
        alpha_init = np.zeros((y_train.shape[0]), dtype=float)
        
        def fun(alpha):
            X = np.einsum("ij,kj->ik", x_train, x_train)
            A = np.einsum("i,j->ij", y_train*alpha, y_train*alpha)
            AX = np.einsum("ij,ji->i", A, X)
            
            return 0.5*np.sum(AX) - np.sum(alpha)
        
        cons=({'type': 'eq',
            'fun': lambda a: np.dot(a.transpose(),y_train)})
        
        bnds = ((0, C),) * alpha_init.shape[0]
        res = optimize.minimize(fun, alpha_init, method='SLSQP', bounds=bnds, constraints=cons)
        
        for i in range(len(y_train)):
            w += res.x[i]*y_train[i]*x_train[i]
        
        idx = (res.x>0) & (res.x<C)
        yb = y_train[idx]
        xb = x_train[idx]
        b = np.mean(yb - np.einsum('i,ji->j', w, xb))
        w = np.append(w, b)
        train_cost = cost(y_train, x_train, w)
        print(f"Final weights={w} for C={C}")
        test_cost = cost(y_test, x_test, w)
        print(f"Error for C={C} : Train={train_cost}; Test={test_cost}")
        print(f"Length of support vectors={np.where(res.x>0)[0].shape[0]}")
    
    # import pickle
    # error_dict = {"costs":costs, "test_cost":test_cost, "lr":r, "w":w}
    # with open('q4b_out.pkl', 'wb') as f:
    #     pickle.dump(error_dict, f)
    
        