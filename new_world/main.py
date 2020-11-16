import argparse
from typing import TypedDict
import pandas as pd
from new_world.utils import Utils
from new_world.network import Network
import numpy as np
from knnclassifier import KnnClassifier



#-n network_rede1.txt -w initial_weights_rede1.txt -f house-votes-84.tsv -s \t -c target
#-n network_rede1.txt -w initial_weights_rede1.txt -f dataset_rede1.txt
#-n network_rede2.txt -w initial_weights_rede2.txt -f dataset_rede2.txt

class CustomArgs(TypedDict):
  filename: str
  k_folds: int
  separator: str
  seed: int
  class_column: str
  network_file: str
  weights_file: str

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Args to run NeuralNetwork")
  parser.add_argument("-f", required=True, dest="filename", help="The dataset filename", type=str)
  # parser.add_argument("-k", required=False, dest="k_folds", default=10, help="Number of folds", type=int)
  parser.add_argument("-c", required=False, dest="class_column", help="Column that has the classification", type=str)
  parser.add_argument("-s", required=False, dest="separator", default=";", help="The data separator", type=str)
  parser.add_argument("-n", required=True, dest="network_file", help="Network file: provides the topology of the network", type=str)
  parser.add_argument("-w", required=False, dest="weights_file", default="", help="Weights File", type=str)
  # parser.add_argument("-seed", required=False, dest="seed", default=26, help="Seed to random", type=int)

  args: CustomArgs = parser.parse_args()
  print("args.weights_file", args.weights_file)
  if len(args.weights_file) != 0:
    weights = Utils.read_weights(args.weights_file)
  else:
    weights = None
  network_topology, regulatizarion_fac = Utils.read_network_file(args.network_file)

  file_name: str = args.filename

  if file_name.endswith('.txt'):
    dataframe: pd.DataFrame = Utils.text_to_dataframe(file_name)
  else:
    dataframe: pd.DataFrame = pd.read_csv("./assets/" + file_name, sep=args.separator)
    dataframe, class_dictionary = Utils.get_xy_dataframe(dataframe, args.class_column)


  filter_col_x = [col for col in dataframe if col.startswith('x')]
  filter_col_y = [col for col in dataframe if col.startswith('y')]

  x_df = dataframe[filter_col_x]
  x_df = ((x_df - x_df.min()) / (x_df.max() - x_df.min()))

  y_df = dataframe[filter_col_y]

  x_matrix = np.matrix(x_df.to_numpy())
  y_matrix = np.matrix(y_df.to_numpy())

  number_of_layers = len(network_topology)

  # the dataset will aways define the number of layers that we will have in the first and last layer, so I dont care for the configuration file
  network_topology[0] = x_matrix.shape[1]
  network_topology[-1] = y_matrix.shape[1]

  neural = Network(number_of_layers, x_matrix, y_matrix, regulatizarion_fac, network_weights=weights, network_topology=network_topology)
  neural.backpropagation()
  print("cost", neural.cost_function())
  neural.calculate_gradient_numerical_verification()
  neural.compare_gradients_with_numerical_estimation()

  #############################################
  ##     Validação K-Cross Estrafificada     ##
  #############################################

  # class_column = GET FROM ARGS
  # k = GET FROM ARGS
  # k_folds = GET K FOLDS

  ## KnnClassifier.k_fold_cross_validation(k, dataframe, class_column, neural)


