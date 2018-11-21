import numpy as np
import math
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# KMeans function
def Kmeans(X, k):
    # Arbitrally select k examples from X as the starting mu
    mu_start = np.random.choice(np.arange(X.shape[0]), size = k, replace = False)
    mu = np.array([])
    for i in range(k):
        if not mu.any():
            mu = X[i].reshape(1,-1)
        else:
            mu = np.append(mu,X[i].reshape(1,-1), axis = 0)
    change = 1
    c = [-1] * X.shape[0]
    d = np.zeros((X.shape[0], k))
    count = 0
    while change > 0.01:
        # Assign points to clusters
        clusters = {}
        for i in range(X.shape[0]):
            max_distance = math.inf
            for j in range(k):
                d[i,j] = math.sqrt(np.sum((X[i,:] -  mu[j,:])**2))
                if d[i,j] < max_distance:
                    max_distance = d[i,j]
                    c[i] = j
            if c[i] not in clusters.keys():
                clusters[c[i]] = X[i].reshape(1,-1)
            else:
                clusters[c[i]] = np.append(clusters[c[i]],X[i].reshape(1,-1), axis = 0)

        # Move mus to cluster centroids
        mu_old = np.copy(mu)
        for key in clusters.keys():
            for i in range(mu.shape[1]):
                mu[key, i] = np.mean(clusters[key][:,i])

        # Calculate change in centroid position
        change = 0
        for i in range(k):
            change += math.sqrt(np.sum((mu[i,:] -  mu_old[i,:])**2))
    return mu, c

# Generate Data
n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

# Group and plot data
mu,colors = Kmeans(X, 3)
plt.figure(figsize=(12,12))
plt.scatter(X[:,0], X[:,1], c=colors)
plt.show()
