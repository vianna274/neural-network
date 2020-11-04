import os
import pandas as pd

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

  @staticmethod
  def text_to_dataframe(filename):
    d = []
    with open('assets/' + filename) as file:
      for line in file.readlines():
        line = line.replace(" ", "")
        line = line.replace("\n", "")
        x, y = line.split(';')
        x = [float(i) for i in x.split(',')]
        y = [float(i) for i in y.split(',')]
        num_input_neurons = len(x)
        num_output_neurons = len(y)
        d.append(x + y)

    my_df = pd.DataFrame(d, columns=['x'+str(i) for i in range(num_input_neurons)]+['y'+str(i) for i in range(num_output_neurons)])
    return my_df




  
          