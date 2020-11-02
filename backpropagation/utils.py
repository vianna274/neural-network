import os

class Utils:

  @staticmethod
  def read_weights(filename):
    weights = []

    filepath = os.path.realpath(os.path.join(os.getcwd(), "assets/" + filename))
    with open(filepath, 'r') as f:
      for line in f:
        formatted_line = line.replace(',', ';').split(';')
        line_weight = []
        for weight in formatted_line:
          line_weight.append(float(weight))
        weights.append(line_weight)
    return weights

  @staticmethod
  def read_network_file(filename):
    regularization_fac = None
    neurons_count = []

    filepath = os.path.realpath(os.path.join(os.getcwd(), "assets/" + filename))
    with open(filepath, 'r') as f:
      for neuron in f:
        if regularization_fac == None:
          regularization_fac = float(neuron)
        else:
          neurons_count.append(int(neuron))
    return neurons_count, regularization_fac
          