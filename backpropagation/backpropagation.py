from typing import List, TypedDict
import numpy as np
import pandas

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
    self.active_values: List[List[int]] = self.get_initial_active_values()
    self.gradients: List[List[int]] = self.get_initial_gradients()
    self.deltas: List[List[float]] = self.get_initial_deltas()
    self.inputs = df[df.columns[pandas.Series(df.columns).str.startswith('x')]]
    self.outputs = df[df.columns[pandas.Series(df.columns).str.startswith('y')]]

    # self.inputs=(inputs-inputs.min())/(inputs.max()-inputs.min())
    if (weights == None):
      self.weights = self.get_random_weights()
    else:
      self.weights = weights

    testInput = self.inputs.iloc[0].values.tolist()
    output = self.outputs.iloc[0].values.tolist()
    out_values = self.propagate(testInput)
    self.update_deltas(output)
    self.update_weights(testInput)
    j_value = self.j_function(out_values, output)

  def update_weights(self, inputs):
    temp_inputs = inputs.copy()
    temp_inputs.insert(0, BIAS_VALUE)

    for neuron in range(0, self.get_num_of_neurons_by_layer(0)+BIAS):
      for weight in range(0, self.get_num_of_neuron_next_layer(0)):
        active_value = temp_inputs[neuron]
        delta = self.deltas[0][weight]
        gradient = active_value * delta
        print("Layer",str(0),"Neuron", neuron, "Weight", weight, "gradient", gradient, "active_value", active_value, "delta", delta)

    for layer in range(1, self.num_of_layers-1):
      for neuron in range(0, self.get_num_of_neurons_by_layer(layer)+BIAS):
        for weight in range(0, self.get_num_of_neuron_next_layer(layer)):
          active_value = self.active_values[layer][neuron]
          delta = self.deltas[layer][weight]
          gradient = active_value * delta
          print("Layer",layer,"Neuron", neuron, "Weight", weight, "gradient", gradient, "active_value", active_value, "delta", delta)
          # TODO: Setar o peso correto
    print("=====")

  def update_deltas(self, correct_values):
    for neuron in range(0, self.get_num_of_neurons_by_layer(self.num_of_layers-1)):
      neuron_active_value = self.active_values[self.num_of_layers-1][neuron]
      last_delta_layer = self.num_of_layers-2
      erro = neuron_active_value - correct_values[neuron]
      self.deltas[last_delta_layer][neuron] = erro
      print("Layer", str(self.num_of_layers-1), "Neuron", neuron, "Delta=", erro)

    for layer in range(self.num_of_layers-2, 0, -1):
      for neuron in range(0, self.get_num_of_neurons_by_layer(layer)):
        erro = 0.0
        for weights in range(1, self.get_num_of_neuron_next_layer(layer)+BIAS):
          weight_value = self.get_neuron_weight(layer, neuron)
          delta_value = self.deltas[layer][weights-BIAS]
          active_value = self.active_values[layer][neuron+BIAS]
          erro += weight_value * delta_value
        erro = erro * active_value * (1 - active_value)
        print("Layer", layer, "Neuron", neuron, "Delta=", erro)
        self.deltas[layer-1][neuron] = erro
    print("=====")
  
  def get_neuron_weights(self, layer, neuron):
    weights = []
    previous_neurons = self.get_num_of_neuron_previous_layer(layer) + BIAS

    for i in range(0, previous_neurons):
      offset = neuron * previous_neurons
      weights.append(self.weights[layer-1][i + offset])
    return weights

  def propagate(self, inputs):
    temp_inputs = inputs.copy()
    temp_inputs.insert(0, BIAS_VALUE)

    for layer in range(1, self.num_of_layers):
      new_inputs = []
      for neuron in range(0, self.get_num_of_neurons_by_layer(layer)):
        weights = self.get_neuron_weights(layer, neuron)
        active_value = self.activate(weights, temp_inputs)
        output = self.sigmoid(active_value)
        if (layer == self.num_of_layers-1):
          self.active_values[layer][neuron] = output
        else:
          self.active_values[layer][neuron+BIAS] = output
          
        print("Layer", layer, "Neuron", neuron, "Active_value", output)
        new_inputs.append(output)
      temp_inputs = new_inputs
      if (layer != self.num_of_layers-1):
        temp_inputs.insert(0,BIAS_VALUE)
    print("=====")
    return temp_inputs

  def j_function(self, out_values, predicted_values):
    cost = 0.0
    for out_value, predicted in zip(out_values, predicted_values):
      # Aqui é a regulamentação
      cost += ((-predicted * (np.log(out_value))) - ((1 - predicted) * np.log(1 - out_value)))
    return float(cost/len(out_values))

  def activate(self, weights: List[float], inputs: List[float]):
    activation = 0.0

    for i in range(0, len(weights)):
      # print ("Peso " + str(weights[i]) + " vezes " + str(inputs[i]))
      activation += weights[i] * inputs[i]
    return activation

  def get_num_of_neuron_next_layer(self, layer):
    return self.neurons[layer+1]

  def get_num_of_neuron_previous_layer(self, layer):
    return self.neurons[layer-1]

  def get_num_of_neurons_by_layer(self, layer):
    return self.neurons[layer]

  def sigmoid(self, activation):
	  return 1.0 / (1.0 + np.exp(-activation))

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

    values = np.zeros(self.get_num_of_neurons_by_layer(self.num_of_layers-1), dtype=float)
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

  def get_neuron_weight(self, layer, neuron):
    return self.weights[layer][neuron+BIAS]