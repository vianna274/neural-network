import argparse
from typing import TypedDict
import pandas as pd
from backpropagation.utils import Utils
from backpropagation.backpropagation import BackPropagation

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
  parser.add_argument("-k", required=False, dest="k_folds", default=10, help="Number of folds", type=int)
  parser.add_argument("-c", required=True, dest="class_column", help="Class to be predicted", type=str)
  parser.add_argument("-s", required=False, dest="separator", default=";", help="The data separator", type=str)
  parser.add_argument("-n", required=True, dest="network_file", help="Network file", type=str)
  parser.add_argument("-w", required=False, dest="weights_file", default="", help="Weights File", type=str)
  parser.add_argument("-seed", required=False, dest="seed", default=26, help="Seed to random", type=int)

  args: CustomArgs = parser.parse_args()
  weights = Utils.read_weights(args.weights_file)
  neurons_count, regulatizarion_fac = Utils.read_network_file(args.network_file)

  file_name: str = args.filename

  if file_name.endswith('.txt'):
    dataframe: pd.DataFrame = Utils.text_to_dataframe(file_name)
  else:
    dataframe: pd.DataFrame = pd.read_csv("./assets/" + file_name, sep=args.separator)
  neural = BackPropagation(neurons_count, regulatizarion_fac, dataframe, weights)
  


