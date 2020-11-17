import pandas as pd
import numpy as np


class KFolds:
    def __init__(self, dataframe, y_set, k, filter_col_y):
        self.df = dataframe
        self.y_set = y_set
        self.k = k
        self.filter_col_y = filter_col_y

    """
      Returns k lists of lists that represent the folds
    """
    def get_folds(self):
      folds = [[] for i in range(self.k)]
      data_shuffled = self.df.sample(frac=1)

      for column in self.filter_col_y:
        values_sets = np.array_split(data_shuffled[data_shuffled[column] == 1], 10)

        for i in range(self.k):
          folds[i].append(values_sets[i])

      for i in range(len(folds)):
        folds[i] = pd.concat(folds[i]).sample(frac=1)

      return folds

    @staticmethod
    def join_k_folds_excluding_one(k_folds:list, excluded_index):
        new_k_folds = []
        index = 0

        for k_fold in k_folds:
            if index != excluded_index:
                new_k_folds.append(k_fold)
            index += 1

        return pd.concat(new_k_folds)