import numpy as np
import matplotlib.pyplot as plt

def find_closest_centroids(X, centroids):
    K = centroids.shape[0]
    m = X.shape[0]
    idx = np.zeros(m, dtype=int)

    for i in range(m):
        distance = []
        for j in range(K):
            dist = np.linalg.norm(X[i]-centroids[j])
            distance.append(dist)
        idx[i] = np.argmin(distance)
    
    return idx 

def compute_centroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))
    counts = np.zeros((m))
    for i in range(m):
        counts[idx[i]]+=1
        centroids[idx[i]]+= X[i]
    for i in range(K):
        centroids[i]/=counts[i]

    return centroids

def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):
    m,n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    idx = np.zeros(m)
    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X,idx,K)
        print(i,"/",max_iters)
    return centroids,idx 
original_img = plt.imread('bird_small.png')
plt.imshow(original_img)

def kMean_init_centroids(X,K):
    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[:K]]
    return centroids
#reshape to mx3, m is the shape of current matrix, and 3 is RGB
X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))

K = 16
max_iters = 10
initial_centroids = kMean_init_centroids(X_img, K)
final_centroids, final_idxs = run_kMeans(X_img, initial_centroids, max_iters)

X_recovered = final_centroids[final_idxs,:]

X_recovered = np.reshape(X_recovered,original_img.shape)

fig, ax = plt.subplots(1,2, figsize=(16,16))
plt.axis('off')

ax[0].imshow(original_img)
ax[0].set_title('Original')
ax[0].set_axis_off()


# Display compressed image
ax[1].imshow(X_recovered)
ax[1].set_title('Compressed with %d colours'%K)
ax[1].set_axis_off()
plt.show()