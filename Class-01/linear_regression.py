import copy, math
import numpy as np 
import matplotlib.pyplot as plt 
#plt.use('./deeplearning.mplstyle')
np.set_printoptions(precision=2)

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

def predict(x,w,b):
    p = np.dot(x,w) +b
    return p

x_vec = X_train[0,:]
f_wb = predict(x_vec,w_init,b_init)
print("Initial prediction: ", f_wb)

def compute_cost(X,y,w,b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i],w)+b
        cost += (f_wb_i-y[i])**2
    cost /=2
    return cost

cost = compute_cost(X_train, y_train, w_init, b_init)
print('Cost at optimal w : {cost}')

def compute_gradient(X,y,w,b):
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0
    # dj_dw  stores change/gradient in each w's [w1, w2, w3...wn]
    for i in range(m):
        # finding the total error compares to a point
        err = np.dot(X[i],w)+b-y[i]
        # finding and adding gradient of each w's
        for j in range(n):
            dj_dw[j] += X[i,j]*err
        dj_db +=err
    dj_dw/=m
    dj_db/=m
    return dj_dw,dj_db
def gradient_descent(X,y,w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        dw, db = gradient_function(X,y,w,b)
        w = w- alpha*dw
        b = b - alpha * db
        if i<10000:
            J_history.append(cost_function(X,y,w,b))
        if i% 100 == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
    return w, b, J_history


initial_w = np.zeros_like(w_init)
initial_b = 0
iterations = 100000
alpha = 5.0e-7
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
# plotting the graph
fig, (ax1,ax2) = plt.subplots(1,2, constrained_layout=True, figsize=(2,4))
ax1.plot(J_hist)
ax2.plot(100+np.arange(len(J_hist[100:])),J_hist[100:])
ax1.set_title("Cost vs iteration"); ax2.set_title("cost vs iteration(tail)")
ax1.set_ylabel("cost"); ax2.set_ylabel("cost")
ax1.set_xlabel("iteration step"); ax2.set_xlabel("iteration step")
plt.show()