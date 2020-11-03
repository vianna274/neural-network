from typing import List, TypedDict
import numpy as np
from math import exp

BIAS = 1
BIAS_VALUE = 1.0

class Neuron(TypedDict):
  gradient: float
  active_value: float
  delta: float
  in_weights: List[float]

class BackPropagation:
  def __init__(self, neurons: List[int], regularization_param: int, weights: List[List[float]] = None):
    self.neurons = neurons
    self.num_of_layers = len(neurons)
    self.regularization_param = regularization_param
    self.active_values: List[List[int]] = self.get_initial_active_values()
    self.gradients: List[List[int]] = self.get_initial_gradients()
    self.deltas: List[List[float]] = self.get_initial_deltas()

    if (weights == None):
      self.weights = self.get_random_weights()
    else:
      self.weights = weights

    self.propagate([0.13, 0.9])

  def get_neuron_weights(self, layer, neuron):
    weights = []
    previous_neurons = self.get_num_of_neuron_previous_layer(layer) + BIAS
    for i in range(0, previous_neurons):
      offset = neuron * previous_neurons
      weights.append(self.weights[layer-1][i + offset])
    return weights

  def set_neuron_weights(self, layer, neuron, weights):
    curr_neurons = self.get_num_of_neurons_by_layer(layer)
    for i in range(0, curr_neurons):
      offset = neuron * (i + 1)
      weights[layer + 1][i + offset]

  def propagate(self, inputs):
    temp_inputs = inputs
    for layer in range(1, self.num_of_layers-1):
      new_inputs = []
      for neuron in range(0, self.get_num_of_neurons_by_layer(layer)):
        weights = self.get_neuron_weights(layer, neuron)
        active_value = self.activate(weights, temp_inputs)
        print("active " + str(active_value))
        output = self.sigmoid(active_value)
        print("out " + str(output))
        new_inputs.append(output)
      temp_inputs = new_inputs
    return temp_inputs

  def activate(self, weights: List[float], inputs: List[float]):
    activation = 0.0

    for i in range(0, len(weights)):
      print ("Peso " + str(weights[i]) + " vezes " + str(inputs[i]))
      activation += weights[i] * inputs[i]
    return activation

  def get_num_of_neuron_next_layer(self, layer):
    return self.neurons[layer+1]

  def get_num_of_neuron_previous_layer(self, layer):
    return self.neurons[layer-1]

  def get_num_of_neurons_by_layer(self, layer):
    return self.neurons[layer]

  def sigmoid(self, activation):
	  return 1.0 / (1.0 + exp(-activation))

  def get_random_weights(self):
    weights = []

    for i in range(1, self.num_of_layers):
      curr_num_of_neurons = self.get_num_of_neurons_by_layer(i)
      previous_num_of_neurons = self.get_num_of_neuron_previous_layer(i) + BIAS
      weights.append(np.random.normal(size=(curr_num_of_neurons * previous_num_of_neurons)))
    return weights

  def get_initial_gradients(self):
    gradients = []
    for layer in range(0, self.num_of_layers - 1):
      next_neurons = self.get_num_of_neuron_next_layer(layer)
      current_neurons = self.get_num_of_neurons_by_layer(layer) + BIAS
      current_gradients = np.zeros(next_neurons * current_neurons, dtype=float)
      gradients.append(current_gradients)
    return gradients

  def get_initial_active_values(self):
    active_values = []

    for layer in range(0, self.num_of_layers-1):
      num_of_neurons = self.get_num_of_neurons_by_layer(layer) + BIAS
      next_num_of_neurons = self.get_num_of_neuron_next_layer(layer)
      values = np.zeros(num_of_neurons * next_num_of_neurons, dtype=float)
      for idx in range(0, next_num_of_neurons):
        offset = (num_of_neurons -1) * idx
        values[idx + offset] = BIAS_VALUE
      active_values.append(values)
    return active_values

  def get_initial_deltas(self):
    deltas = []

    for layer in range(1, self.num_of_layers):
      num_of_neurons = self.get_num_of_neurons_by_layer(layer)
      values = np.zeros(num_of_neurons, dtype=float)
      deltas.append(values)
    return deltas

  def get_neuron_info(self, layer, neuron): 
    gradient = self.gradients[layer][neuron]
    active_value = self.active_values[layer][neuron]
    
    if (layer == 0):
      in_weights = []
    else:
      in_weights = []
      previous_neurons = self.get_num_of_neuron_previous_layer(layer)
      for i in range(0, previous_neurons):
        offset = neuron * previous_neurons
        in_weights.append(self.weights[layer][i + offset])

    if (neuron == 0): # Se for o bias não tem
      delta = 0
    else:
      delta = self.deltas[layer][neuron]

    neuron_info: Neuron = Neuron()
    neuron_info['gradient'] = gradient
    neuron_info['active_value'] = active_value
    neuron_info['in_weights'] = in_weights
    neuron_info['delta'] = delta
    return neuron_info