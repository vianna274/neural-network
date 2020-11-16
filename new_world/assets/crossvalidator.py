from kfolds import KFolds
from network import Network
import pandas as pd
import numpy as np
import time
import statistics

class CrossValidator:

    def __init__(self, k, data, target_class, number_of_layers, regularization_fac, weights, network_topology):
        self.k = k
        self.df = data
        self.target_class = target_class
        self.number_of_layers = number_of_layers
        self.regularization_fac = regularization_fac
        self.weights = weights
        self.network_topology = network_topology

    def k_fold_cross_validation(self):
        results_for_each_permutation = []
        accuracies = []
        classify_time_mean = 0
        train_time_mean = 0

        # Transformar os dados em KFolds
        kfolds = KFolds(self.df, self.target_class, self.k)
        k_folds = KFolds.get_folds()

        start_geral_time = time.time()

        # Iterativamente treinar um modelo:
        for i in range(self.k):
            test_fold_index = i
            # Utilizando k-1 folds de treino e 1 de teste, variando o fold de teste a cada repetição.
            training_k_folds = KFolds.join_k_folds_excluding_one(k_folds, test_fold_index)
            test_k_fold = k_folds[test_fold_index]

            # valores de x e y de treino
            x = training_k_folds.drop(self.target_class, axis=1).values
            y = pd.get_dummies(training_k_folds[self.target_class]).values

            # valores de x e y de teste
            x_test = test_k_fold.drop(self.target_class, axis=1).values
            y_test = pd.get_dummies(test_k_fold[self.target_class]).values

            # Criação da rede e Treinamento
            model = Network(self.number_of_layers, x, y, self.regularization_fac, self.weights, self.network_topology)
            model.train()

            # Coletar resultados da predição utilizando o fold de teste
            start_time = time.time()
            result = model.classify(test_k_fold)
            end_time = time.time()
            classify_time_mean += (end_time - start_time)

            # Os resultados desta iteração são adicionados na lista:
            results_for_each_permutation.append(result)

            # TODO: n sei direito o que faz esse trecho:
            correct = 0
            for i, r in enumerate(result):
                if test_k_fold.iloc[[i]][self.target_class].values[0] == r:
                    correct += 1

            accuracies.append(correct / len(result))

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

        return result


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
