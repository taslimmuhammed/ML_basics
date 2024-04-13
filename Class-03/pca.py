import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 

X = np.array([[ 99,  -1],
       [ 98,  -1],
       [ 97,  -2],
       [101,   1],
       [102,   1],
       [103,   2]])
pca_2 = PCA(n_components=2)
pca_2.fit(X)
print(pca_2.explained_variance_ratio_)
X_trans_2 = pca_2.transform(X)
print(X_trans_2)
X_reduced_2 = pca_2.inverse_transform(X_trans_2)
print(X_reduced_2)