from typing import List
import numpy as np

class BackPropagation:

  def __init__(self, neurons: List[int], regularization_param: int, weights: List[List[int]] = None):
    self.neurons = neurons
    self.num_of_layers = len(neurons)
    self.regularization_param = regularization_param

    if (weights == None):
      self.weights = self.get_random_weights()
    else:
      self.weights = weights

  def get_random_weights(self):
    weights = []

    for i in range(1, self.num_of_layers):
      curr_num_of_neurons = self.neurons[i]
      previous_num_of_neurons = self.neurons[i - 1] + 1
      weights.append(np.random.normal(size=(curr_num_of_neurons * previous_num_of_neurons)))
    return weights