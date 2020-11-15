import numpy as np
from layer import Layer
from typing import List

BIAS = 1


class Network:
  def __init__(self, number_of_layers: int, x: np.matrix, y: np.matrix, epslon: float):
    self.number_of_layers = number_of_layers
    self.fx = None
    self.epslon: float = epslon

    # x and y represents the values for x and y from the training set
    self.x: np.matrix = x
    self.y: np.matrix = y

    # current_x and current y are the ones used for the calculation in one example
    self.current_x: np.matrix = None
    self.current_y: np.matrix = None
    self.number_of_examples = len(y)

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

    # TODO: this block of layer creation needs to be extracted to a method and we need to create it from the file
    layer1 = Layer(1, 2, loaded_weights_matrix=weights1)
    layer2 = Layer(2, 1, loaded_weights_matrix=weights2)
    layer3 = Layer(1, 0)
    self.layers: List[Layer] = [layer1, layer2, layer3]

    ## real implementation
    # self.fx = self.propagate()
    # self.cost = self.cost_function()
    # self.print_network_information()
    print(self.x)
    print(self.y)

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
      print("J do exemplo " + str(index + 1) + ": " + str(examples_costs[index]))
    j = (sum(examples_costs))/n #3 divides the total error by the number of examples
    s = self.get_regularization_factor() #4&5  calculates the regularization term

    return j + s # returns the regularized cost

  def get_regularization_factor(self):
    s = 0  # not bias weights sum
    n = self.number_of_examples

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

      self.print_network_information()

      self.layers[last_layer_index].set_delta(self.fx - self.current_y) # 1.2 calculates the delta for the last layer

      print("delta ", last_layer_index+1, self.fx - self.current_y)

      for k in range(last_hidden_layer_index, first_hidden_layer_index, -1): # 1.3 calculates delta for the hidden layers
        self.layers[k].calculate_delta(self.layers[k+1])
        self.layers[k].remove_first_element_from_delta()
        print("delta ", k + 1, self.layers[k].delta)

      for k in range(last_hidden_layer_index, first_layer_index, -1): # 1.4 for each layer, updates the gradients based in the current example
        print("k", k)
        self.layers[k].update_gradients(self.layers[k+1])

    for k in range(last_hidden_layer_index, first_layer_index, -1):  # 2
      self.layers[k].calculate_final_gradients(n)
      print("Gradientes finais", k+1, self.layers[k].D)


    for k in range(last_hidden_layer_index, first_layer_index, -1):  # 3 in the end of the epoch we update the weights
      self.layers[k].update_weights()

    return "Hey back propagation working"







if __name__ == '__main__':
  np.random.seed(4)
  x = np.matrix([[0.13], [0.42]])
  y = np.matrix([[0.9], [0.23]])
  network = Network(3, x=x, y=y, epslon=0.0)

  print(network.backpropagation())



# multiplicar a matrix com os pesos dos neuroneos pela matrix das entradas
# calcular o g da matrix

# matrix theta layer 1 (1-2) onde 1 é a origem (número de neuronios na camada de destino por número de pesos entrando em cada neuronio)
# vetor com entradas = xi (iésima instancia de treinamento).
# multiplicação é a matrix z2 que segnifica
# aplicar a gradiente para pegar as ativações para ter o vetor a de camada de destino
