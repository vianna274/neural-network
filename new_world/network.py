import numpy as np
from layer import Layer
from typing import List
import pandas as pd

BIAS = 1

class Network:
  def __init__(self, number_of_layers: int, x: np.matrix, y: np.matrix, regularization_factor: float, network_weights: List, network_topology: List, debug_flag: bool, alpha, stop_criteria):
    self.number_of_layers = number_of_layers
    self.fx = None
    self.regularization: float = regularization_factor
    self.debug_flag = debug_flag
    self.network_topology = network_topology
    self.alpha = alpha
    # x and y represents the values for x and y from the training set
    self.x: np.matrix = x
    self.y: np.matrix = y

    # current_x and current y are the ones used for the calculation in one example
    self.current_x: np.matrix = None
    self.current_y: np.matrix = None
    self.number_of_examples = len(y)
    self.variation = 0.000001

    self.network_final_gradients_per_layer = [] #used for the numerical verification
    self.network_estimated_gradients_per_layer = [] #used for the numerical verification

    self.layers: List[Layer] = []
    network_weights = self.handle_inputed_weights(network_weights)
    self.initialize_network_weights(network_topology, network_weights, number_of_layers)
    self.stop_criteria = stop_criteria

  def handle_inputed_weights(self, network_weights):
    if network_weights is None:
      return self.generate_initial_weights()
    else:
      return network_weights

  def generate_initial_weights(self):
    generated_weights = []
    for k in range(len(self.network_topology)-1):
      generated_weights.append(np.matrix(np.random.random([self.network_topology[k+1], self.network_topology[k] + BIAS])))
    return generated_weights

  def initialize_network_weights(self, network_topology, network_weights, number_of_layers):
    for k in range(number_of_layers):
      if k + 1 <= number_of_layers - 1:
        self.layers.append(
          Layer(network_topology[k], network_topology[k + 1], self.regularization, loaded_weights_matrix=network_weights[k], debug_flag=self.debug_flag, alpha=self.alpha))
      else:
        self.layers.append(Layer(network_topology[k], 0, self.regularization, debug_flag=self.debug_flag,  alpha=self.alpha))

  def propagate(self): # for 1 - n-1
    self.layers[0].neuron_values = np.transpose(self.current_x) # sets the value for the neurons in the first layer
    self.layers[0].add_bias_neuron()

    for k in range(1, self.number_of_layers - 1):
      self.layers[k].propagate(self.layers[k-1], is_last_layer=False)
    self.layers[self.number_of_layers-1].propagate(self.layers[self.number_of_layers-2], is_last_layer=True)

    return self.layers[self.number_of_layers-1].neuron_values

  def cost_function(self):
    """
      Implements the cost function (J)
    :return:
    """
    n = self.number_of_examples  # n = number of examples in the training set

    examples_costs = [] #1 initialize the variable that will be accumulating the total error for the network
    for index, example in enumerate(zip(self.x, self.y)): # for each example (x(i), y(i)) in the training set

      # configures the inputs  and the outputs for the network for each example
      self.current_x = example[0]
      self.current_y = example[1]

      fx = self.propagate() #2.1 propagate x(i) and get the fθ(x(i)) outputs predicted by the network

      # 2.2 calculates the vector J(i) (for the example) with the associated cost for each output from the network from the current example
      examples_costs.append((-self.current_y) * np.log(fx) - ((1 - self.current_y) * np.log(1 - fx)))
    j = (sum(examples_costs))/n #3 divides the total error by the number of examples
    s = self.get_regularization_factor() #4&5  calculates the regularization term

    return j + s # returns the regularized cost

  def get_regularization_factor(self):
    s = 0  # not bias weights sum
    n = self.number_of_examples

    for layer in self.layers:
      s += np.sum(np.power(layer.get_not_bias_weights(), 2))
    s = (self.regularization / (2 * n)) * s

    return s

  def print_network_information(self):
    print("\nprinting network information: ")
    for k in range(self.number_of_layers):
      print("----------")
      print("Layer: " + str(k + 1))
      if (self.layers[k].z_matrix is not None):
        print("z matrix: ")
        print(self.layers[k].z_matrix.squeeze(1))
      print("activations: ")
      print(self.layers[k].neuron_values.squeeze(1))
    print("f(x)", self.layers[self.number_of_layers-1].neuron_values.squeeze(1))


  def backpropagation(self):
    """
      Implements the back propagation in a vectorized way
      implementation
    """

    n = self.number_of_examples
    last_layer_index = self.number_of_layers - 1
    first_layer_index = -1
    last_hidden_layer_index = last_layer_index - 1
    first_hidden_layer_index = 0

    for example in zip(self.x, self.y): #1 for each example in the training set

      # configures the inputs  and the outputs for the network for each example
      self.current_x = example[0]
      self.current_y = example[1]

      self.fx = self.propagate() # 1.1

      if (self.debug_flag):
        self.print_network_information()

      self.layers[last_layer_index].set_delta(self.fx - np.transpose(self.current_y)) # 1.2 calculates the delta for the last layer

      if (self.debug_flag):
        print("delta ", last_layer_index+1, self.fx - np.transpose(self.current_y))

      for k in range(last_hidden_layer_index, first_hidden_layer_index, -1): # 1.3 calculates delta for the hidden layers
        self.layers[k].calculate_delta(self.layers[k+1])
        self.layers[k].remove_first_element_from_delta()
        if (self.debug_flag):
          print("delta ", k + 1, self.layers[k].delta)

      for k in range(last_hidden_layer_index, first_layer_index, -1): # 1.4 for each layer, updates the gradients based in the current example
        if (self.debug_flag):
          print("k", k)
        self.layers[k].update_gradients(self.layers[k+1])

    for k in range(last_hidden_layer_index, first_layer_index, -1):  # 2
      self.layers[k].calculate_final_gradients(n)
      self.network_final_gradients_per_layer.append(self.layers[k].D)
      if (self.debug_flag):
        print("Gradientes finais", k+1, self.layers[k].D)

    if (self.debug_flag):
      self.calculate_gradient_numerical_verification()
      self.compare_gradients_with_numerical_estimation()

    for k in range(last_hidden_layer_index, first_layer_index, -1):  # 3 in the end of the epoch we update the weights
      self.layers[k].update_weights()

    return "Hey back propagation working"

  def calculate_gradient_numerical_verification(self):
    """
      Calculates estimation for the gradients in the network and stores it to network_estimated_gradients
    """

    last_layer_index = self.number_of_layers - 1
    first_layer_index = -1
    last_hidden_layer_index = last_layer_index - 1

    network_estimated_gradients = []

    print("\nPrinting the numerical validation for the gradients")
    for k in range(last_hidden_layer_index, first_layer_index, -1):  # 1 in the end of the epoch we update the weights
      original_theta = np.copy(self.layers[k].weights_matrix)
      layer_estimated_gradients = []
      for i in range(original_theta.shape[0]):
        line = []
        for j in range(original_theta.shape[1]):
          # calculates the positive variation
          self.layers[k].weights_matrix = np.copy(original_theta)
          self.layers[k].weights_matrix[i][j] += self.variation
          positive_variation_cost = float(self.cost_function()[0][0])

          # calculates the negative variation
          self.layers[k].weights_matrix = np.copy(original_theta)
          self.layers[k].weights_matrix[i][j] -= self.variation
          negative_variation_cost = float(self.cost_function()[0][0])
          line.append((positive_variation_cost - negative_variation_cost)/(2 * self.variation))
        layer_estimated_gradients.append(line)
      if (self.debug_flag):
        print("Numerical gradient aprox for layer ", k, layer_estimated_gradients)
      network_estimated_gradients.append(layer_estimated_gradients)

    self.network_estimated_gradients_per_layer = [np.matrix(x) for x in network_estimated_gradients]

  def compare_gradients_with_numerical_estimation(self):
    last_layer_index = self.number_of_layers - 1
    first_layer_index = -1
    last_hidden_layer_index = last_layer_index - 1

    if (self.debug_flag):
      print("estimated", self.network_estimated_gradients_per_layer)
      print("calculated", self.network_final_gradients_per_layer)

    for k in range(last_hidden_layer_index, first_layer_index, -1):
      layer_numerical_estimation: np.matrix = np.array(self.network_estimated_gradients_per_layer[k])
      layer_final_calculated: np.matrix = np.array(self.network_final_gradients_per_layer[k])

      mean_diff = np.mean(np.abs(layer_numerical_estimation - layer_final_calculated))
      if (self.debug_flag):
        print('Erro médio entre grandiente via backprop e grandiente numerico para Theta%d: %.10f' % (k + 1, mean_diff))

  def train(self):
    cost_list = []
    criteria_not_reached = True
    while criteria_not_reached:
      previous_cost = self.cost_function()
      cost_list.append(previous_cost.item(0))
      self.backpropagation()
      current_cost = self.cost_function()

      criteria_not_reached = abs(current_cost - previous_cost) > self.stop_criteria

    return cost_list
  def classify_dataset(self, dataset):
    """
      Classifies an dataset
    :param dataset: pd.Dataframe
    :return: list of results
    """

    instances = np.matrix(dataset.to_numpy())
    results = []

    for i in range(instances.shape[0]):
      results.append(self.classify(instances[i]))

    return results

  def classify(self, instance: np.matrix):
    """
      Classifies an instance
    :param instance: np.matrix
    :return: the result of the propagation
    """
    self.current_x = instance
    return self.propagate()