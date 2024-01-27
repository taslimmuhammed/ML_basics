import numpy as np
import matplotlib.pyplot as plt
import copy
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])  

# fig, ax = plt.subplots(1,1,figsize=(4,4))
# ax.plot(X_train)
# ax.axis([0,4,0,3.5])
# ax.set_ylabel("X1", fontsize=12)
# ax.set_xlabel("X2", fontsize=12)
# plt.show()
def sigmoid(z):
    return 1/(1+np.exp(z))

def logistic_cost(X,y,w,b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i], w)+b
        f_wb_i = sigmoid(z_i)
        cost += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
    
    cost/=m
    return cost

w_tmp = np.array([1,1])
b_tmp = -3
print(logistic_cost(X_train, y_train, w_tmp, b_tmp))

def compute_logistic_gradient(X,y,w,b):
    m,n, = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        z_i = np.dot(X[i],w)+b
        f_wb_i = sigmoid(z_i)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += err_i*X[i,j]
        dj_db += err_i

    dj_db /= m
    dj_dw /=m
    return dj_dw, dj_db

def gradient_descent_logostic(X,y,w_in, b_in,alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = compute_logistic_gradient(X,y,w,b)
        w-=alpha*dj_dw
        b-=alpha*dj_db
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( logistic_cost(X, y, w, b) )

    return w,b,J_history

w_tmp  = np.zeros_like(X_train[0])
b_tmp  = 0.
alph = 0.1
iters = 10000

w_out, b_out, _ = gradient_descent_logostic(X_train, y_train, w_tmp, b_tmp, alph, iters) 
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")
    