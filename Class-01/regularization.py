import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1+np.exp(z))

def cost_linear(X,y,w,b,lambda_=1):
    m = X.shape[0]
    n = len(w)
    cost=0
    for i in range(m):
        f_wb_i = np.dot(X[i],w)+b
        cost+= (f_wb_i-y[i])**2
    cost /= (2*m)

    reg_cost = 0
    for i in range(n):
        reg_cost += (w[j]**2)
    reg_cost*=(lambda_/2**m)
    return cost+reg_cost

def cost_logistic(X,y,w,b,lambda_=1):
    m,n = X.shape
    cost = 0
    for i in range(m):
        z = np.dot(X[i],w)+b
        f_wb_i = sigmoid(z)
        cost+= -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
    cost/=m
    reg_cost=0
    for j in range(n):
        reg_cost += w[j]**2
    reg_cost*=(lambda_/(2**m))
    return reg_cost+cost
# Gradient of the linear function
def gradient_linear(X,y,w,b,lambda_):
    m,n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0
    # Adding MSE part
    for i in range(m):
        f_wb_i = np.dot(X[i],w)+b
        err = f_wb_i-y[i]
        for j in range(n):
            dj_dw[i]+=err*X[i,j]
        dj_db+=err
    dj_db/=m
    dj_dw/=m
    # Adding Regularization part
    for j in range(n):
        dj_dw[j] += lambda_*w[j]/m
    return dj_dw, dj_db

# similar to linear
def gradient_logistic(X,y,w,b,lambda_):
    m,n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0
    for i in range(m):
        z_i = np.dot(X[i],w)+b
        f_wb_i = sigmoid(z_i)
        err = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += err*X[i,j]
        dj_db+=err
    dj_db/=m
    dj_dw/=m
    for j in range(n):
        dj_dw[j] += lambda_*w[j]/(m)
    return dj_dw, dj_db
