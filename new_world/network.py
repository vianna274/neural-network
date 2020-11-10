import numpy as np
from layer import Layer

BIAS = 1


class Network:
  def __init__(self, number_of_layers: int, y: np.array, epslon: float):
    self.number_of_layers = number_of_layers
    self.fx = None
    self.epslon: float = epslon
    self.y: np.array = y
    # 0.40000, 0.10000; 0.30000, 0.20000
    # 0.70000, 0.50000, 0.60000

    entries = np.matrix([0.13])

    weights1 = np.matrix(
      [
        [0.4, 0.1],
        [0.3, 0.2]
      ]
    )

    weights2 = np.matrix(
      [
        [0.7, 0.5, 0.6]
      ]
    )

    entries = np.transpose(entries)
    layer1 = Layer(1, 2, neuron_values=entries, loaded_weights_matrix=weights1)
    layer2 = Layer(2, 1, loaded_weights_matrix=weights2)
    layer3 = Layer(1, 0)

    # 0.13000; 0.90000 primeiro exemplo
    # 0.42000; 0.23000 segundo

    self.layers = [layer1, layer2, layer3]

    ## real implementation
    self.fx = self.propagate()
    self.cost = self.cost_function()
    self.print_network_information()

  def propagate(self): # for 1 - n-1
    self.layers[0].add_bias_neuron()

    for k in range(1, self.number_of_layers - 1):
      self.layers[k].propagate(self.layers[k-1], is_last_layer=False)
    self.layers[self.number_of_layers-1].propagate(self.layers[self.number_of_layers-2], is_last_layer=True)

    return self.layers[self.number_of_layers-1].neuron_values

  def cost_function(self):
    """
      Implements the cost function
    :return:
    """
    n = len(self.y)
    j = (-self.y) * np.log(self.fx) - ((1 - self.y) * np.log(1 - self.fx))
    j = j.sum()/n

    s = self.get_regularization_factor()

    return j + s

  def get_regularization_factor(self):
    s = 0  # not bias weights sum
    n = len(self.y)

    for layer in self.layers:
      s += np.sum(np.power(layer.get_not_bias_weights(), 2))
    s = (self.epslon/(2*n)) * s

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
    print("J do exemplo 1: ", self.cost)


if __name__ == '__main__':
  np.random.seed(4)
  y = np.array([0.9])
  network = Network(3, y, epslon=0.0)


# multiplicar a matrix com os pesos dos neuroneos pela matrix das entradas
# calcular o g da matrix

# matrix theta layer 1 (1-2) onde 1 é a origem (número de neuronios na camada de destino por número de pesos entrando em cada neuronio)
# vetor com entradas = xi (iésima instancia de treinamento).
# multiplicação é a matrix z2 que segnifica
# aplicar a gradiente para pegar as ativações para ter o vetor a de camada de destino
