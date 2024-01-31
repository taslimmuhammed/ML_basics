import numpy as np 
def g(z):
    return 1/(1+np.exp(z))
x = np.array([200,17])
W = np.array([
    [1,-3,5],
    [-2,4,-6] 
    ]) #the w vector for a neuron exist in a column not row
b = np.array([-1,1,2])

def dense_loop(a_in, W,b):
    units = W.shape[1] #no of nuerons
    a_out = np.array(units)
    for j in range(units):
        w = W[:,j] # gives output as [1,-2] etc for each neuron
        z = np.dot(w,a_in)+b[j]
        a_out[j] = g(z)
    return a_out
def dense_vect(A_in,W,B): # vectors are represented in capital
    Z = np.matmul(A_in,W)+B #matrix multiplication
    A_out = g(Z)
    return A_out

