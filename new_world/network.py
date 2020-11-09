import numpy as np

class Layer:
  def __init__(self, current_number_of_neurons: int, next_number_of_neurons: int, loaded_weights_matrix: np.matrix = None, neuron_values: np.matrix = None):
    self.current_number_of_neurons = current_number_of_neurons
    self.next_number_of_neurons = next_number_of_neurons
    # setup layer weights
    self.origin_destiny_weights_matrix: np.matrix = None
    self.setup_initial_weights(loaded_weights_matrix)
    self.neuron_values: np.matrix = None # vai ser uma matrix coluna
    self.z_matrix: np.matrix = None # matrix coluna que contém os valores q, se aplicado sigmoind, vai resultar na ativação da próxima layer
  def setup_initial_weights(self, loaded_weights_matrix):
    """
      Setup the initial weights for the network
      If we have something loaded, we use it, if not, we create the weights randomly
    """
    if loaded_weights_matrix is None:
      self.origin_destiny_weights_matrix = np.random.random_sample(
        [self.next_number_of_neurons, self.current_number_of_neurons])
    else:
      self.origin_destiny_weights_matrix = loaded_weights_matrix


class Network:
  def __init__(self):
    entries = np.matrix([1,2,3])
    entries = np.transpose(entries)
    layer1 = Layer(3, 4, entries=entries)
    layer2 = Layer(4, 4)

    self.get_next_layer_activations(layer1, layer2)

  def get_next_layer_activations(self, origin: Layer, destiny: Layer):
    print(origin.origin_destiny_weights_matrix)
    print(origin.neuron_values)
    z_matrix: np.matrix = np.dot(origin.origin_destiny_weights_matrix, origin.neuron_values)
    print(z_matrix)

if __name__ == '__main__':
  np.random.seed(4)
  network = Network()


# multiplicar a matrix com os pesos dos neuroneos pela matrix das entradas
# calcular o g da matrix

# matrix theta layer 1 (1-2) onde 1 é a origem (número de neuronios na camada de destino por número de pesos entrando em cada neuronio)
# vetor com entradas = xi (iésima instancia de treinamento).
# multiplicação é a matrix z2 que segnifica
# aplicar a gradiente para pegar as ativações para ter o vetor a de camada de destino
