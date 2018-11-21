import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''Activation Funtions and their derivatives'''
# Relu Activation
def relu(Z):
    A = np.maximum(0,Z)
    return A

def d_relu(Z):
    A = relu(Z)
    dG = (A != 0) * 1
    return dG

# Sigmoid activation
def sigmoid(Z):
    A = 1/(1 + np.exp(-Z))
    return A

def d_sigmoid(Z):
    A = sigmoid(Z)
    dG = A * (1 - A)
    return dG

# Hyperbolic tangent (tanh) activation
def tanh(Z):
    A = (2 / (1 + np.exp(-2 * Z))) - 1
    return A

def d_tanh(Z):
    A = tanh(Z)
    dG = 1 - A**2
    return dG












mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
