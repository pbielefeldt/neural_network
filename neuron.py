#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# activation function
# the sigmoid function maps (−∞,+∞) to (0,1)
def sigmoid(x):
  return 1 / (1+np.exp(-x))

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