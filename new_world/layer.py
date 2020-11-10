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

  @staticmethod
  def sigmoid(z):
    return 1. / (1. + np.exp(-z))