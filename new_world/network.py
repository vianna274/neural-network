import numpy as np
from .layer import Layer
from typing import List

BIAS = 1


class Network:
  def __init__(self, number_of_layers: int, x:np.array, y: np.array, epslon: float):
    self.number_of_layers = number_of_layers
    self.fx = None
    self.epslon: float = epslon

    # x and y represents the values for x and y from the training set
    self.x = np.array = x
    self.y: np.array = y

    # current_x and current y are the ones used for the calculation in one example
    self.current_x = None
    self.current_y = None

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

    self.layers: List[Layer] = [layer1, layer2, layer3]

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


  def backpropagation(self):
    """
      Implements the back propagation in a vectorized way
      implementation
    """

    n = len(x)
    last_layer_index = self.number_of_layers - 1
    first_layer_index = -1
    last_hidden_layer_index = last_layer_index - 1
    first_hidden_layer_index = 0

    for example in zip(x,y): #1 for each example in the training set

      # configures the inputs  and the outputs for the network for each example
      self.current_x = example[0]
      self.current_y = example[1]

      self.fx = self.propagate() # 1.1


      self.layers[last_layer_index].set_delta(self.fx - self.y) # 1.2 calculates the delta for the last layer

      for k in range(last_hidden_layer_index, first_hidden_layer_index, -1): # 1.3 calculates delta for the hidden layers
        self.layers[k].calculate_delta(self.layers[k+1])
        self.layers[k].remove_first_element_from_delta()

      for k in range(last_hidden_layer_index, first_layer_index, -1): # 1.4 for each layer, updates the gradients based in the current example
        self.layers[k].update_gradients(self.layers[k+1])

    for k in range(last_hidden_layer_index, first_layer_index, -1):  # 2
      self.layers[k].calculate_final_gradients(n)

    for k in range(last_hidden_layer_index, first_layer_index, -1):  # 3
      self.layers[k].update_weights()







if __name__ == '__main__':
  np.random.seed(4)
  x = np.array([0.9])
  y = np.array([0.9])
  network = Network(3, x=x, y=y, epslon=0.0)


# multiplicar a matrix com os pesos dos neuroneos pela matrix das entradas
# calcular o g da matrix

# matrix theta layer 1 (1-2) onde 1 é a origem (número de neuronios na camada de destino por número de pesos entrando em cada neuronio)
# vetor com entradas = xi (iésima instancia de treinamento).
# multiplicação é a matrix z2 que segnifica
# aplicar a gradiente para pegar as ativações para ter o vetor a de camada de destino
