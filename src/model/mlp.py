
import numpy as np

# from util.activation_functions import Activation
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier
from util.activation_functions import Activation
from report.evaluator import Evaluator

class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, input_weights=None,
                 output_task='classification', output_activation='softmax',
                 cost='crossentropy', learning_rate=0.01, epochs=50):

        """
        A digit-7 recognizer based on logistic regression algorithm

        Parameters
        ----------
        train : list
        valid : list
        test : list
        learning_rate : float
        epochs : positive int

        Attributes
        ----------
        training_set : list
        validation_set : list
        test_set : list
        learning_rate : float
        epochs : positive int
        performances: array of floats
        """

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.output_task = output_task  # Either classification or regression
        self.output_activation = output_activation
        self.cost = cost

        self.training_set = train
        self.validation_set = valid
        self.test_set = test

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        self.layers = layers
        self.input_weights = input_weights

        # add bias values ("1"s) at the beginning of all data sets
        self.training_set.input = np.insert(self.training_set.input, 0, 1,
                                            axis=1)
        self.validation_set.input = np.insert(self.validation_set.input, 0, 1,
                                              axis=1)
        self.test_set.input = np.insert(self.test_set.input, 0, 1, axis=1)

        # Build up the network from specific layers
        # Here is an example of a MLP acting like the Logistic Regression
        self.layers = []
        self.layers.append(LogisticLayer(784, 10, None, "sigmoid", True))
        self.layers.append(LogisticLayer(10, 2, None, "softmax", False))

    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self.get_layer(0)

    def _get_output_layer(self):
        return self.get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """
        layer1Out = self.layers[0].forward(inp)
        layer1Out = np.insert(layer1Out, 0, 1)
        layer2Out = self.layers[1].forward(layer1Out)

        return layer2Out

    def _compute_error(self, target):
        """
        Compute the total error of the network

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """

        print("LOLOL", self.layers[-1].outp)
        self.layers[-1].computeDerivative(np.array(target - self.layers[-1].outp), np.array(1.0))
        return self.layers[-1].deltas

    def _update_weights(self):
        """
        Update the weights of the layers by propagating back the error
        """
        pass

    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        evaluator = Evaluator()

        for epoch in range(self.epochs):
            print("Training Epoch", epoch)
            for i, input, label in zip(range(len(self.training_set.input)), self.training_set.input, self.training_set.label):
                out = self._feed_forward(input)
                error = self._compute_error(label).reshape(1, 1)
                for layer, nextLayer in zip(self.layers[:-1], self.layers[1:]):
                    weights = nextLayer.weights[1:].T
                    layer.computeDerivative(error, weights)

                for layer in self.layers:
                    layer.updateWeights(self.learning_rate)

            evaluator.printAccuracy(self.test_set, self.evaluate())

    def _train_one_epoch(self):
        """
        Train one epoch, seeing all input instances
        """

        pass

    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here

        outp = self._feed_forward(test_instance)
        return outp > 0.5

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.test_set.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.training_set.input = np.delete(self.training_set.input, 0, axis=1)
        self.validation_set.input = np.delete(self.validation_set.input, 0,
                                              axis=1)
        self.test_set.input = np.delete(self.test_set.input, 0, axis=1)
