import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from lab_utils_multi import  load_house_data

X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']

scaler = StandardScaler()
X_norm  = scaler.fit_transform(X_train)

sgdr = SGDRegressor(max_iter=1000)
sgdr.fit_transform(X_norm, y_train)

b_norm = sdgr.intercept_
w_norm = sdgr.coef_

