import numpy as np
import matplotlib.pyplot as plt

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