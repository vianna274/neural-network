from kfolds import KFolds
from network import Network
from printtofile import PrintToFile
import pandas as pd
import numpy as np
import time
import statistics
from typing import List

class CrossValidator:

    def __init__(self, k, data, filter_col_y, y_matrix, number_of_layers, regularization_fac, weights, network_topology, alpha, stop_criteria):
        self.k = k
        self.df = data
        self.filter_col_y = filter_col_y
        self.number_of_layers = number_of_layers
        self.regularization_fac = regularization_fac
        self.weights = weights
        self.network_topology = network_topology
        self.y_matrix = y_matrix
        self.alpha = alpha
        self.stop_criteria = stop_criteria

    def k_fold_cross_validation(self):
        results_for_each_permutation = []
        accuracies = []
        classify_time_mean = 0
        train_time_mean = 0
        iteration_costs = []

        # Transformar os dados em KFolds
        k_folds = KFolds(self.df, self.y_matrix, self.k, self.filter_col_y)
        folds = k_folds.get_folds()

        start_geral_time = time.time()

        # Iterativamente treinar um modelo:
        for i in range(self.k):
            test_fold_index = i
            # Utilizando k-1 folds de treino e 1 de teste, variando o fold de teste a cada repetição.
            training_k_folds = folds[:]
            training_k_folds.pop(test_fold_index)
            training_k_folds = pd.concat(training_k_folds)

            test_k_fold = folds[test_fold_index]

            # valores de x e y de treino
            filter_col_x = [col for col in training_k_folds if col.startswith('x')]
            filter_col_y = [col for col in training_k_folds if col.startswith('y')]

            x_df = training_k_folds[filter_col_x]
            x_df = ((x_df - x_df.min()) / (x_df.max() - x_df.min()))
            y_df = training_k_folds[filter_col_y]

            x_matrix = np.matrix(x_df.to_numpy())
            y_matrix = np.matrix(y_df.to_numpy())

            test_k_fold_without_y = test_k_fold[filter_col_x]
            test_k_fold_y = test_k_fold[filter_col_y]

            self.network_topology[0] = x_matrix.shape[1]
            self.network_topology[-1] = y_matrix.shape[1]

            # Criação da rede e Treinamento
            model = Network(self.number_of_layers, x_matrix, y_matrix, self.regularization_fac, self.weights, self.network_topology, debug_flag=False, alpha=self.alpha, stop_criteria=self.stop_criteria)
            iteration_costs.append(model.train())

            # Coletar resultados da predição utilizando o fold de teste
            start_time = time.time()
            result = model.classify_dataset(test_k_fold_without_y)
            end_time = time.time()
            classify_time_mean += (end_time - start_time)

            accuracy = CrossValidator.compare_predicted_with_real(test_k_fold_y, result)

            # Os resultados desta iteração são adicionados na lista:
            results_for_each_permutation.append(result)

            accuracies.append(accuracy)

        end_geral_time = time.time()
        geral_time = end_geral_time - start_geral_time

        result = {}
        result['classification_mean_time'] = classify_time_mean / len(accuracies)
        result['train_mean_time'] = train_time_mean / len(accuracies)
        result['geral_time'] = geral_time
        result['mean_accuracy'] = statistics.mean(accuracies)
        result['std_accuracy'] = statistics.stdev(accuracies)
        result['min'] = min(accuracies)
        result['max'] = max(accuracies)

        PrintToFile.print_j(iteration_costs)
        PrintToFile.print_accuracy(accuracies)
        PrintToFile.print_statistics(result)

        return result


    @staticmethod
    def compare_predicted_with_real(complete: pd.DataFrame, predicted_results: List):
        predicted_correctness = []
        corrects = 0
        for i, predicted_answer in enumerate(predicted_results):
            ans = np.transpose(predicted_answer).tolist()[0]
            real_ans = complete.iloc[[i]].values[0].tolist()

            predicted_class = ans.index(max(ans))
            real_class = real_ans.index(max(real_ans))
            if predicted_class == real_class:
                corrects += 1

        return corrects/len(predicted_results)

    ##### N SEI SE VAI PRECISAR DISSO ABAIXO
    @staticmethod
    def get_results(model, test_k_fold, target_class, target_class_values):
        conf_matrix = CrossValidator.get_conf_matrix(model, test_k_fold, target_class, target_class_values)

        acc = CrossValidator.get_accuracy(conf_matrix, target_class_values)

        print("Accuracy:", acc)

        return {
            'accuracy': acc
        }

    @staticmethod
    def get_conf_matrix(model, test_k_fold, target_class, target_class_values):
        matrix = {}

        ## Inicialização da confusion matrix
        for c in  target_class_values:
            matrix[c] = {}

            for predicted in target_class_values:
                matrix[c][predicted] = 0

        for i, row in test_k_fold.iterrows():
            _class = row[target_class]
            predicted = None

            matrix[_class][predicted] = matrix[_class][predicted] + 1

        return matrix

    @staticmethod
    def get_accuracy(conf_matrix, target_class_values):
        total = 0
        accuracy = 0

        for value in target_class_values:
            accuracy = accuracy + conf_matrix[value][value]

            for index in target_class_values:
                total = total + conf_matrix[value][index]

        return accuracy / total
