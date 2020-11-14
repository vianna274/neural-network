import numpy as np

BIAS = 1


class Layer:
  def __init__(self, current_number_of_neurons: int, next_number_of_neurons: int, loaded_weights_matrix: np.matrix = None, neuron_values: np.matrix = None):
    self.current_number_of_neurons = current_number_of_neurons
    self.next_number_of_neurons = next_number_of_neurons
    # setup layer weights
    self.weights_matrix: np.matrix = None
    self.neuron_values: np.matrix = neuron_values # vai ser uma matrix coluna
    self.setup_initial_weights(loaded_weights_matrix)
    self.z_matrix: np.matrix = None # matrix coluna que contém os valores q, se aplicado sigmoind, vai resultar na ativação da próxima layer
    self.delta = None
    self.D = None
    self.alpha = 0.0

  def propagate(self, previous_layer, is_last_layer):
    self.z_matrix = np.dot(previous_layer.weights_matrix, previous_layer.neuron_values)
    self.neuron_values = Layer.sigmoid(self.z_matrix)

    if not is_last_layer:
      self.add_bias_neuron()

  def add_bias_neuron(self):
    self.neuron_values = np.insert(self.neuron_values, 0, BIAS, axis=0)

  def get_not_bias_weights(self):
    return np.delete(self.weights_matrix, 0, 1)

  def setup_initial_weights(self, loaded_weights_matrix):
    """
      Setup the initial weights for the network
      If we have something loaded, we use it, if not, we create the weights randomly
    """
    if loaded_weights_matrix is None:
      self.weights_matrix = np.random.random_sample(
        [self.next_number_of_neurons, self.current_number_of_neurons])
    else:
      self.weights_matrix = loaded_weights_matrix

  def set_layer_delta(self, delta):
    """
      Sets the delta value for the layer
      (designed to set the delta for the last layer by the back propagation)
    :param delta:
    :return:
    """
    self.delta = delta

  def calculate_delta(self, next_layer):
    """
      Calculate the deltas for the layer based on the next layer
    :param next_layer:
    """
    self.delta = np.multiply(np.multiply(np.transpose(self.weights_matrix) * next_layer.delta, self.neuron_values), (1 - self.neuron_values))
    self.delta = np.delete(self.delta, 0, 1) #TODO: verify the axis

  def update_gradients(self, next_layer):
    """
      This function prepares the gradients (not final version) based on the weights and delta
    :param next_layer:
    """
    self.D = self.D + np.dot(next_layer.delta, np.transpose(self.neuron_values))

  def calculate_final_gradients(self, n):
    """
      Calculate the regularized gradients
    :param n: number of examples
    """
    P = self.regularization * self.weights_matrix
    self.D = (1/n) * (self.D + P)

  def update_weights(self):
    """
      This function needs to be called once we have the final gradients for the layer
      It will update the weights based on the calculated gradients
    """
    self.weights_matrix = self.weights_matrix - (self.alpha * self.D)

  @staticmethod
  def sigmoid(z):
    return 1. / (1. + np.exp(-z))