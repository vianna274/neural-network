import pandas as pd


class KFolds:
    def __init__(self, dataframe, y_set, k, class_column):
        self.df = dataframe
        self.y_set = y_set
        self.k = k
        self.class_column = ['y0', 'y1']

    """
      Returns k lists of lists that represent the folds
    """
    def get_folds(self):
      folds_indexes = []
      fold_index = 0

      for i in range(0, self.k):
        folds_indexes.append([])

      for value in self.possible_values():
        subset_data_value = self.df[self.df[self.class_column] == value]

        for index in subset_data_value.index.values:
            folds_indexes[fold_index].append(index)
            fold_index += 1

            if fold_index == len(folds_indexes):
                fold_index = 0

      return self.get_folds_from_dataframe(folds_indexes)

    def possible_values(self):
        class_values = self.df[self.class_column]
        hash = {}
        unique_class_values = []

        for class_value in class_values.values:
            if str(class_value) not in hash.keys():
                hash[str(class_value)] = True
                unique_class_values.append(class_value)

        return unique_class_values


    def get_folds_from_dataframe(self, folds_indexes):
        folds = []

        for indexes in folds_indexes:
            folds.append(self.df[self.df.index.isin(indexes)])

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