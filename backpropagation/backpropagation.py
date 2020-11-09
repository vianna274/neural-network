from typing import List, TypedDict
from copy import deepcopy
import numpy as np
import pandas
from copy import deepcopy

BIAS = 1
BIAS_VALUE = 1.0

class Neuron(TypedDict):
  gradient: float
  active_value: float
  delta: float
  in_weights: List[float]

class BackPropagation:
  def __init__(self, neurons: List[int], regularization_param: int, df, weights: List[List[float]] = None):
    self.neurons = neurons
    self.num_of_layers = len(neurons)
    self.regularization_param = regularization_param
    self.active_values = self.get_initial_active_values()
    self.deltas = self.get_initial_deltas()
    self.gradients = self.get_initial_gradients()
    self.inputs = df[df.columns[pandas.Series(df.columns).str.startswith('x')]]
    self.outputs = df[df.columns[pandas.Series(df.columns).str.startswith('y')]]
    self.numeric_evaluation = True
    print(self.gradients)
    # TODO: Transformar em um atributo a ser recebido
    self.change_value = 0.2
    # TODO: Fazer ser opcional a normalizacao dos inputs
    # self.inputs=(inputs-inputs.min())/(inputs.max()-inputs.min())

    if (weights == None):
      self.weights = self.get_random_weights()
    else:
      self.weights = weights
    testInput = self.inputs.iloc[0].values.tolist()
    output = self.outputs.iloc[0].values.tolist()

    self.propagate(testInput)
    out = self.active_values[self.num_of_layers-1][1:]
    print(self.active_values[self.num_of_layers-1][1:])
    j_value = self.j_function(out, output)
    print("JValue=", j_value)
    self.update_deltas(output)
    self.update_gradients()
    self.update_weights(j_value)

  def update_weights(self, j_value):
    for layer_idx in range(1, self.num_of_layers):
      for neuron_idx in range(0, self.get_num_of_neurons_by_layer(layer_idx)):
        for weight in range(0, self.get_num_of_neuron_next_layer(layer_idx)):
          print(self.gradients)
          print(self.weights)
          old_weight = self.get_neuron_weight_by_weight(layer_idx, neuron_idx+1, weight)
          new_weight = old_weight - (self.change_value * j_value * self.gradients[layer_idx][neuron_idx][weight])
          
          print("old_weight", old_weight)
          print("new_weight", new_weight)

  def update_gradients(self): 
    for neuron_idx in range(0, self.get_num_of_neurons_by_layer(0, bias=True)):
      for weight_idx in range(0, self.get_num_of_neuron_next_layer(0, bias=False)):
        active_value = self.get_active_value_by_neuron(0, neuron_idx)
        # print(self.deltas)
        delta = self.get_neuron_delta(1, weight_idx+1)
        gradient = active_value * delta
        print("Gradient=", gradient, "Layer", 0, "Neuron", neuron_idx, "weight", weight_idx, "delta", delta, "active_value", active_value)
        # print(self.gradients)
        self.gradients[1][weight_idx][neuron_idx] = gradient

    for layer in range(1, self.num_of_layers-1):
      for neuron_idx in range(0, self.get_num_of_neurons_by_layer(layer, bias=True)):
        for weight_idx in range(0, self.get_num_of_neuron_next_layer(layer, bias=False)):
          active_value = self.get_active_value_by_neuron(layer, neuron_idx)
          delta = self.get_neuron_delta(layer+1, weight_idx+1)
          gradient = active_value * delta

          print("Gradient=", gradient, "Layer", layer, "Neuron", neuron_idx, "weight", weight_idx, "delta", delta, "active_value", active_value)
          # print(self.gradients)
          # print(self.gradients[layer+1])
          self.gradients[layer+1][weight_idx][neuron_idx] = gradient
          # TODO: Setar o peso correto
    print("=====")

  def j_function(self, out_values, predicted_values):
    cost = 0.0
    print("out_values", out_values)
    print("predicted_values", predicted_values)
    for out_value, predicted in zip(out_values, predicted_values):
      # Aqui é a regulamentação FORA GOVERNO
      cost += ((-predicted * (np.log(out_value))) - ((1 - predicted) * np.log(1 - out_value)))
    return cost

  def update_deltas(self, correct_values):
    last_layer = self.num_of_layers-1
    for neuron in range(1, self.get_num_of_neurons_by_layer(last_layer, bias=True)):
      neuron_active_value =  self.get_active_value_by_neuron(last_layer, neuron)
      erro = neuron_active_value - correct_values[neuron-1]
      self.change_delta_value_by_neuron(last_layer, neuron, erro)
      print("Delta=", erro, "Layer", last_layer, "Neuron", neuron)

    for layer in range(self.num_of_layers-2, 0, -1):
      for neuron in range(1, self.get_num_of_neurons_by_layer(layer, bias=True)):
        erro = 0.0
        for next_neuron in range(1, self.get_num_of_neuron_next_layer(layer, bias=True)):
          weight_value = self.get_neuron_weight_by_weight(layer+1, next_neuron, neuron)
          delta_value = self.get_neuron_delta(layer + 1, next_neuron)
          erro += weight_value * delta_value
        active_value = self.get_active_value_by_neuron(layer, neuron)
        erro = erro * active_value * (1 - active_value)
        print("Delta=", erro, "Layer", layer, "Neuron", neuron)
        self.change_delta_value_by_neuron(layer, neuron, erro)
    print("=====")

  def get_active_value_by_neuron(self, layer, neuron):
    return self.active_values[layer][neuron]

  def get_active_values_by_layer(self, layer):
    return self.active_values[layer]

  def change_weight_by_weight_index(self, layer, neuron, weight_idx, new_value):
    self.weights[layer][neuron][weight_idx] = new_value

  def change_delta_value_by_neuron(self, layer, neuron, new_value):
    self.deltas[layer][neuron] = new_value

  def change_layer_active_value_by_neuron(self, layer, neuron, new_value):
    self.active_values[layer][neuron] = new_value

  def change_layer_active_values(self, layer, active_values):
    self.active_values[layer] = active_values

  def propagate(self, inputs):
    temp_inputs = deepcopy(inputs)
    temp_inputs.insert(0, BIAS_VALUE)

    self.change_layer_active_values(0, temp_inputs)

    for layer in range(1, self.num_of_layers-1):
      for neuron in range(1, self.get_num_of_neurons_by_layer(layer, bias=True)):
        weights = self.get_neuron_in_weights(layer, neuron)
        previous_active_values = self.get_active_values_by_layer(layer-1)
        active_value = self.activate(previous_active_values, weights)
        normalized_active_value = self.sigmoid(active_value)
        # print("active_value", normalized_active_value, "layer", layer, "neuron", neuron)
        self.change_layer_active_value_by_neuron(layer, neuron, normalized_active_value)

    last_layer = self.num_of_layers-1
    for neuron in range(1, self.get_num_of_neurons_by_layer(self.num_of_layers-1, bias=True)):
      weights = self.get_neuron_in_weights(last_layer, neuron)
      previous_active_values = self.get_active_values_by_layer(last_layer-1)
      active_value = self.activate(previous_active_values, weights)
      normalized_active_value = self.sigmoid(active_value)
      self.change_layer_active_value_by_neuron(last_layer, neuron, normalized_active_value)
      # print("active_value", normalized_active_value, "layer", last_layer, "neuron", neuron)

    # print("=====")
    return temp_inputs

  def activate(self, weights: List[float], inputs: List[float]):
    activation = 0.0
    for i in range(0, len(weights)):
      activation += weights[i] * inputs[i]
    return activation

  def get_neuron_delta(self, layer, neuron):
    # print("Layer", layer, "Neuron", neuron, "deltas", self.deltas)
    return self.deltas[layer][neuron]

  def get_neuron_out_value(self, layer, neuron, bias=True):
    if (bias):
      return self.active_values[layer][neuron]
    else:
      return self.active_values[layer][neuron+BIAS]

  def get_neuron_weight_by_weight(self, layer, neuron, weight):
    return self.weights[layer][neuron][weight]

  def get_neuron_in_weights(self, layer, neuron):
    return self.weights[layer][neuron]

  def get_initial_active_values(self):
    active_values = []

    for layer in range(0, self.num_of_layers):
      num_of_neurons = self.get_num_of_neurons_by_layer(layer, bias=True)
      values = np.zeros(num_of_neurons, dtype=float).tolist()
      values[0] = BIAS_VALUE
      active_values.append(values)

    return active_values

  def get_random_weights(self):
    weights = []
    weights.append([]) # Primeiro layer nao tem pesos

    for i in range(1, self.num_of_layers):
      layer_weights = []
      previous_num_of_neurons = self.get_num_of_neuron_previous_layer(i, bias=True)

      for _ in range(0, self.get_num_of_neurons_by_layer(i, bias=False)):
        _weights = np.random.normal(size=previous_num_of_neurons)
        _weights.insert(0, [])
        layer_weights.append(weights)
      weights.append(layer_weights)

    return weights
  
  def sigmoid(self, activation):
	  return 1.0 / (1.0 + np.exp(-activation))

  def get_initial_deltas(self):
    deltas = []
    deltas.append([])

    for layer in range(1, self.num_of_layers):
      num_of_neurons = self.get_num_of_neurons_by_layer(layer, bias=False)
      values = np.zeros(num_of_neurons, dtype=float).tolist()
      values.insert(0, 99999)
      deltas.append(values)

    return deltas

  def get_initial_gradients(self):
    gradients = []
    gradients.append([])

    for i in range(1, self.num_of_layers):
      layer_gradients = []
      previous_num_of_neurons = self.get_num_of_neuron_previous_layer(i, bias=True)

      for _ in range(0, self.get_num_of_neurons_by_layer(i, bias=False)):
        _gradients = np.zeros(previous_num_of_neurons).tolist()
        layer_gradients.append(_gradients)
      gradients.append(layer_gradients)

    return gradients

  def get_num_of_neuron_next_layer(self, layer, bias = False):
    return self.get_num_of_neurons_by_layer(layer+1, bias)

  def get_num_of_neuron_previous_layer(self, layer, bias = False):
    return self.get_num_of_neurons_by_layer(layer-1, bias)

  def get_num_of_neurons_by_layer(self, layer, bias = False):
    if bias:
      return self.neurons[layer] + BIAS
    else: 
      return self.neurons[layer]
