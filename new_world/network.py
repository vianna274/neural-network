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
    z = np.dot(previous_layer.weights_matrix, previous_layer.neuron_values)
    self.neuron_values = Layer.sigmoid(z)
    if not is_last_layer:
      self.add_bias_neuron()


  def add_bias_neuron(self):
    self.neuron_values = np.insert(self.neuron_values, 0, BIAS, axis=0)


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

class Network:
  def __init__(self, number_of_layers: int):
    self.number_of_layers = number_of_layers

    # 0.40000, 0.10000; 0.30000, 0.20000
    # 0.70000, 0.50000, 0.60000

    entries = np.matrix([0.13])

    weights1 = np.matrix([[0.4, 0.1],
                        [0.3, 0.2]])

    weights2 = np.matrix([[0.7],
                         [0.5],
                         [0.6]])

    entries = np.transpose(entries)
    layer1 = Layer(1, 2, neuron_values=entries, loaded_weights_matrix=weights1)
    layer2 = Layer(2, 1, loaded_weights_matrix=weights2)
    layer3 = Layer(1, 0)

    # 0.13000; 0.90000 primeiro exemplo
    # 0.42000; 0.23000 segundo

    self.layers = [layer1, layer2, layer3]

    ## real implementation
    self.propagate()

  def propagate(self): # for 1 - n-1
    self.layers[0].add_bias_neuron()
    for k in range(1, self.number_of_layers - 1):
      self.layers[k].propagate(self.layers[k-1], is_last_layer=False)
    self.layers[self.number_of_layers-1].propagate(self.layers[self.number_of_layers-2], is_last_layer=True)

    return self.layers[self.number_of_layers-1].neuron_values

if __name__ == '__main__':
  np.random.seed(4)
  network = Network(3)


# multiplicar a matrix com os pesos dos neuroneos pela matrix das entradas
# calcular o g da matrix

# matrix theta layer 1 (1-2) onde 1 é a origem (número de neuronios na camada de destino por número de pesos entrando em cada neuronio)
# vetor com entradas = xi (iésima instancia de treinamento).
# multiplicação é a matrix z2 que segnifica
# aplicar a gradiente para pegar as ativações para ter o vetor a de camada de destino
