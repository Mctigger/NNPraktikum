#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven

from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from model.mlp import MultilayerPerceptron
from util.activation_functions import Activation
from model.logistic_layer import LogisticLayer

from report.evaluator import Evaluator
from report.performance_plot import PerformancePlot

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d

import numpy as np
import itertools

def main():
    #data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
    #                 one_hot=True, target_digit='7')

    # NOTE:
    # Comment out the MNISTSeven instantiation above and
    # uncomment the following to work with full MNIST task
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
                      one_hot=False)

    # # NOTE:
    # # Other 1-digit classifiers do not make sense now for comparison purpose
    # # So you should comment them out, let alone the MLP training and evaluation
    #
    # # Train the classifiers #
    # print("=========================")
    # print("Training..")
    #
    # # Stupid Classifier
    # myStupidClassifier = StupidRecognizer(data.training_set,
    #                                       data.validation_set,
    #                                       data.test_set)
    #
    # print("\nStupid Classifier has been training..")
    # myStupidClassifier.train()
    # print("Done..")
    # # Do the recognizer
    # # Explicitly specify the test set to be evaluated
    # stupidPred = myStupidClassifier.evaluate()
    #
    # # Perceptron
    # myPerceptronClassifier = Perceptron(data.training_set,
    #                                     data.validation_set,
    #                                     data.test_set,
    #                                     learning_rate=0.005,
    #                                     epochs=10)
    #
    # print("\nPerceptron has been training..")
    # myPerceptronClassifier.train()
    # print("Done..")
    # # Do the recognizer
    # # Explicitly specify the test set to be evaluated
    # perceptronPred = myPerceptronClassifier.evaluate()
    #
    # # Logistic Regression
    # myLRClassifier = LogisticRegression(data.training_set,
    #                                     data.validation_set,
    #                                     data.test_set,
    #                                     learning_rate=0.005,
    #                                     epochs=30)
    #
    # print("\nLogistic Regression has been training..")
    # myLRClassifier.train()
    # print("Done..")
    # # Do the recognizer
    # # Explicitly specify the test set to be evaluated
    # lrPred = myLRClassifier.evaluate()


    # Build up the network from specific layers
    # Here is an example of a MLP acting like the Logistic Regression
    layers = []
    layers.append(LogisticLayer(784, 5, None, "sigmoid", True))
    layers.append(LogisticLayer(5, 10, None, "softmax", False))

    myMLPClassifier = MultilayerPerceptron(data.training_set,
                                           data.validation_set,
                                           data.test_set,
                                           learning_rate=0.5,
                                           epochs=30, layers=layers)
    print("\nLogistic Regression has been training..")
    myMLPClassifier.train()
    print("Done..")
    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    mlpPred = myMLPClassifier.evaluate()
    #
    # Report the result #
    print("=========================")
    evaluator = Evaluator()
    #
    # # print("Result of the stupid recognizer:")
    # # evaluator.printComparison(data.testSet, stupidPred)
    # evaluator.printAccuracy(data.test_set, stupidPred)
    # #
    # # print("\nResult of the Perceptron recognizer (on test set):")
    # # evaluator.printComparison(data.testSet, perceptronPred)
    # evaluator.printAccuracy(data.test_set, perceptronPred)
    # #
    # # print("\nResult of the Logistic Regression recognizer (on test set):")
    # # evaluator.printComparison(data.testSet, perceptronPred)
    # evaluator.printAccuracy(data.test_set, lrPred)
    #
    print("\nResult of the Multi-layer Perceptron recognizer (on test set):")
    # evaluator.printComparison(data.testSet, perceptronPred)
    evaluator.printAccuracy(data.test_set, mlpPred)
    #
    # # Draw
    # plot = PerformancePlot("Logistic Regression")
    # plot.draw_performance_epoch(myLRClassifier.performances,
    #                             myLRClassifier.epochs)

    # 3D Plot learning_rates + epochs -> accuracies
    print("Creating 3D plot. This may take some minutes...")
    learning_rate_sample_count = 5
    epochs_sample_count = 20
    xticks = np.logspace(-10.0, 0, base=10, num=learning_rate_sample_count, endpoint=False)
    accuracies = []
    learning_rates = []
    epoch_values = []

    for i in itertools.product(range(learning_rate_sample_count)):
        learning_rate = 100 / np.exp(i)
        print("Calculating accuracy for: learning rate = %s" % (learning_rate))
        myMLPClassifier = MultilayerPerceptron(data.training_set,
                                               data.validation_set,
                                               data.test_set,
                                               learning_rate=learning_rate,
                                               epochs=epochs_sample_count,
                                               layers=layers)
        epoch_accuracies = myMLPClassifier.train(False)
        lrPred = myMLPClassifier.evaluate()
        epoch_values.append([e for e in range(epochs_sample_count)])
        learning_rates.append([learning_rate for _ in range(epochs_sample_count)])
        accuracies.append(epoch_accuracies)

    accuracies_merged = list(itertools.chain(*accuracies))
    epochs_merged = list(itertools.chain(*epoch_values))
    learning_rates_merged = list(itertools.chain(*learning_rates))
    print(accuracies_merged)
    print(epochs_merged)
    print(learning_rates)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(np.log10(learning_rates_merged), epochs_merged, accuracies_merged)
    ax.set_xlabel("Learning Rate")

    ax.set_xticks(np.log10(xticks))
    ax.set_xticklabels(xticks)
    ax.set_ylabel('Epochs')
    ax.set_zlabel('Accuracy')
    plt.show()

if __name__ == '__main__':
    main()
