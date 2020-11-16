import os
import pandas as pd
import numpy as np

class Utils:

  @staticmethod
  def read_weights(filename):
    """
      Reads the weights file and returns the theta matrix for each layer
    :param filename:
    :return: [theta1, theta2, theta3....]
    """
    filepath = os.path.realpath(os.path.join(os.getcwd(), "assets/" + filename))
    layer_weights = []

    with open(filepath, 'r') as f:
      for line in f:
        layer_weights.append(np.matrix(line))
    return layer_weights

  @staticmethod
  def read_network_file(filename):
    """
      Returns the network topology with the number of neurons per layer
    :param filename:
    :return:
    """
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

  @staticmethod
  def get_xy_dataframe(df: pd.DataFrame, target_column: str):
    """
      Given a dataset, changes the features to xs and outputs to ys

    :param df: dataframe that will be used for training
    :param target_column: the column that will be considered as target for the training
    :return: dataframe with x and y values
    """

    output_df = df

    # treats the X case
    x_columns = [col for col in df if col != target_column]
    x_labels = ["x" + str(i) for i in range(len(x_columns))]
    dictionary = dict(zip(x_columns, x_labels))
    output_df = output_df.rename(columns=dictionary)

    # treats the Y case
    output_df = output_df.drop(columns=[target_column])
    possible_classes = df[target_column].unique()
    y_labels = ["y" + str(i) for i in range(len(possible_classes))]
    class_dictionary = dict(zip(possible_classes, y_labels))
    # add y columns with 0
    for y in y_labels:
      output_df[y] = 0

    number_of_instances = df.shape[0]
    for i in range(number_of_instances):
      instance_class = df.loc[i, target_column]
      output_df.loc[i, [class_dictionary[instance_class]]] = 1

    return output_df, class_dictionary