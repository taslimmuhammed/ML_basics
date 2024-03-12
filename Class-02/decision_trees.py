import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X_train = np.array(
[[1, 1, 1],
 [0, 0, 1],
 [0, 1, 0],
 [1, 0, 1],
 [1, 1, 1],
 [1, 1, 0],
 [0, 0, 0],
 [1, 1, 0],
 [0, 1, 0],
 [0, 1, 0]])
y_train = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])

def entropy(p): #probablity to entropy
    if p==0 or p==1:
        return 0
    else:
        return -p*np.log2(p) - (1-p)*np.log2((1-p))
    
def split_indices(X, feature):
    left_indices = []
    right_indices = []
    for i,x in enumerate(X):
        if x[feature]==1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return left_indices,right_indices

def wighted_entropy(X,y,left_indices,right_indices):
    w_left = len(left_indices)/len(X)
    w_right = len(right_indices)/len(X)
    p_left = sum(y[left_indices])/len(left_indices)
    p_right = sum(y[right_indices])/len(right_indices)
    return  w_left * entropy(p_left) + w_right * entropy(p_right)

def info_gain(X,y,left_indices,right_indices):
    p_root = sum(y)/len(y)
    h_root = entropy(p_root)
    w_entropy = wighted_entropy(X,y,left_indices,right_indices)
    return  h_root - w_entropy

