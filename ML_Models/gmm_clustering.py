import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('figure', figsize=[12,8])
import math
from sklearn.datasets import make_blobs

def prob_of_x_given_gauss(X, mu, sig):
    d = sig.shape[0]
    det_sig = np.linalg.det(sig)
    inv_sig = np.linalg.inv(sig)
    m_distance = ((X - mu)*inv_sig*(X - mu).T)[0,0]
    num = math.exp(-0.5*m_distance)
    denom = math.sqrt(((2*math.pi)**d)*det_sig)
    prob = num/denom
    return prob

def initialize_gauss(X, k):
    # Arbitrarily choose k different Gaussians
    start_pos = np.random.choice(np.arange(X.shape[0]), k, replace = False)
    mu = np.matrix([])                              # Centroids of Gausians
    for pos in start_pos:
        if not mu.any():
            mu = X[pos]
        else:
            mu = np.append(mu, X[pos], axis = 0)
    sig = [np.matrix(np.identity(X.shape[1]))] * k  # Covariance matrices of Gausians
    return mu, sig

def calculate_joint_probabilities(X, mu, sig, phi):
    # Joint probability that Gaussian j was chosen and Xi was sampled
    p_joint= np.matrix(np.zeros((X.shape[0], mu.shape[0])))

    for j in range(mu.shape[0]):
        for i in range(X.shape[0]):
            p_joint[i,j] = prob_of_x_given_gauss(X[i], mu[j], sig[j])*phi[j]
    return p_joint

def step_E(p_joint):
    m = p_joint.shape[0]
    k = p_joint.shape[1]
    c = [-1] * m                        # Point Classification
    p_gen = np.matrix(np.zeros((m, k))) # Probability that exaple was generate by Gaussian
    n = [0] * k                         # Effective points in Gaussian
    for i in range(m):
        max_prob = -1
        for j in range(k):
            p_gen[i,j] = p_joint[i,j]/np.sum(p_joint[i,:])
            n[j] += p_gen[i,j]
            if p_gen[i,j] > max_prob:
                max_prob = p_gen[i,j]
                c[i] = j
    return c, p_gen, n

def step_M(X, p_gen, n):
    m = X.shape[0]
    d = X.shape[1]
    k = len(n)
    # Initiale parameters
    mu = np.matrix(np.zeros((k,d)))
    sig = []
    phi = []

    # Update Parameters
    for j in range(k):
        # Update mu
        for l in range(d):
            mu[j,l] = (p_gen[:,j].T.dot(X[:,l]))[0,0] / n[j]
        # Update Sigma
        offset = (X - mu[j])
        offest_with_prob = np.matrix(np.array(p_gen[:,j]) * np.array(offset))
        cov = (offest_with_prob.T.dot(offset)) / n[j]
        sig.append(cov)
        # Update Phi
        phi.append(n[j]/m)
    return mu, sig, phi

def calculate_change(mu_new, mu_old, sig_new, sig_old):
    k = mu_old.shape[0]
    mu_change = 0
    sig_change = 0
    for j in range(k):
        mu_change += math.sqrt(np.sum(np.array(mu_old[j] - mu_new[j])**2))
        sig_change += math.sqrt(np.sum(np.array(sig_old[j] - sig_new[j])**2))
    change = mu_change + sig_change
    print("Change attributed to Mu: %.3f" % mu_change)
    print("Change attributed to Sigma: %.3f" % sig_change)
    print("Total Change: %.3f" % change)
    print("-------------------------------------")
    return change

def EMClustering(X, k):
    X = np.matrix(X)
    # Arbitrarily choose k different gaussians
    mu, sig = initialize_gauss(X, k)
    phi = [1/k] * k
    ep = 0.01
    change = 1
    while change > ep:
        old_mu = np.copy(mu)
        old_sig = np.copy(sig)
        p_joint = calculate_joint_probabilities(X, mu, sig, phi)
        # Step E
        c, p_gen, n = step_E(p_joint)
        # Step M
        mu, sig, phi = step_M(X, p_gen, n)
        change = calculate_change(mu, old_mu, sig, old_sig)
    return mu, sig, c

# Generate Data
n_samples = 1500
random_state = 170
data_X, data_y = make_blobs(n_samples=n_samples, random_state=random_state)

# Create clusters
r_mu, r_sig, r_c = EMClustering(data_X, 3)

# Plot data
plt.scatter(data_X[:,0], data_X[:,1], c=r_c)
plt.show()
