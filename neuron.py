#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# activation function
# the sigmoid function maps (−∞,+∞) to (0,1)
def sigmoid(x):
  return 1 / (1+np.exp(-x))

# derivative of the sigmoid function
# \frac{\partial f(x)}{\partial x} = f(x) \ast (1-f(x))
def deriv_sigmoid(x):
  f = sigmoid(x)
  return fx * (1 - fx)

# Mean square loss
# mse = \frac{1}{n} \sum_{i=1}^{n}(y_i^\text{true}-y_i^\text{pred})^2
# (note: the ys are np-arrays of same length)
def mse_loss(y_true, y_predict):
  return ((y_true-y_predict)**2).mean()

class Neuron:
  def __init__(self, weights, bias):
    self.weights = weights
    self.bias = bias
  
  # weight input, add bias, apply activation function
  # y = f(x_1*w_1 + x_2*w_2 + b)
  # where f is the sigmoid function
  # (done as a vector product)
  def feedforward(self, input):
    total = np.dot(self.weights, input) + self.bias
    return sigmoid(total)

# w_1 = 0 & w_2 = 1
weights = np.array([0,1])
bias = 4

# let's do the magic …!
#n = Neuron(weights, bias)
#x = np.array([2,3])
#r = n.feedforward(x)
#print(r)

class MyNeuralNetwork:
  '''
  take two inputs, run them through one hidden layer with two neurons, called h1
  and h2, and merge to the output layer with a third neuron, called o1. Each 
  neuron has the same weights and biases, w1=0, w2=1, b=0 for simplicity.
  '''
  def __init__(self):
    # Weights
    self.w1 = np.random.normal()
    self.w2 = np.random.normal()
    self.w3 = np.random.normal()
    self.w4 = np.random.normal()
    self.w5 = np.random.normal()
    self.w6 = np.random.normal()

    # Biases
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()
    
    self.h1 = Neuron(weights, bias)
    self.h2 = Neuron(weights, bias)
    self.o1 = Neuron(weights, bias)
  
  def feedforward(self, x):
    out_h1 = self.h1.feedforward(x)
    out_h2 = self.h2.feedforward(x)
    h = np.array([out_h1, out_h2])
    
    out_o1 = self.o1.feedforward(h)
    return out_o1


a_network = MyNeuralNetwork()
x = np.array([2,3])
r = a_network.feedforward(x)
print(r)

