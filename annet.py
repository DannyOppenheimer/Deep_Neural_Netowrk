import numpy as np
import sys
import re
import pandas as pd
# TODO: Add CNN and RNN functionality


# ************************ Artificial Neural Network ************************
# Instantiate this class to create a fully-connected ANN, most commonly used for classification
# It's high-abstraction functions are Train and Predict
# The model follows this basic path: Input-Validation > Feed-Forward > Back-Propagate
# ****************************************************************************
class ANN:

    # The constructor for the ANN creates the layers, the neurons in those layers, and the connections to those neurons
    def __init__(self, layers, verbose=0, learning_rate=0.1):

        np.random.seed(1)

        self.layers = len(layers)
        self.learning_rate = learning_rate
        self.neurons = []

        self.verbose = verbose
        self.loading_signs = ["[|]", "[/]", "[–]", "[\\]"]

        loop_num = 0
        percentage_total = 0
        percentage_current = 0
        array_lang = " ".join(("Creating array with", str(len(layers)), "layers:"))
        neuron_lang = "Adding neurons to array."
        connection_lang = "Adding connections to each neuron"

        for _ in layers:
            loop_num += 1
            print("\r", " ".join((array_lang, self.loading_signs[loop_num % 4])), end="")
            self.neurons.append([])
        print("\r", " ".join((array_lang, "[✓]\n")))

        for i in range(len(layers)):
            for _ in range(layers[i]):
                loop_num += 1
                percentage_total += 1
                print("\r", " ".join((neuron_lang, self.loading_signs[loop_num % 4])), end="")
                Neuron(i, self.neurons)
        print("\r", " ".join((neuron_lang, "[✓]\n")))

        for i in range(len(self.neurons)):
            for neuron in self.neurons[i]:
                percentage_current += 1
                print("\r", " ".join((connection_lang, str(int(((percentage_current / percentage_total) * 100))), "%",
                                      self.loading_signs[percentage_current % 4])), end="", sep="")
                if i != len(self.neurons) - 1:
                    for forward_neuron in self.neurons[i + 1]:
                        Connection(neuron, forward_neuron)
        print("\r", " ".join((connection_lang, "[✓]\n")))

    # The feed forward function goes forward through each layer, setting each neuron's value to the sigmoid-squashed
    # value from each weight and neuron before it.
    def feed_forward(self, l, inputs):

        for i in range(len(self.neurons[0])):
            self.neurons[0][i].value = inputs[l][i]

        for i in range(len(self.neurons)):
            if i != 0:
                for neuron in self.neurons[i]:
                    temp_value = 0

                    # Using the backward connections for a forward pass might seem counter-intuitive, but it's easier
                    # to add up all the values in the last layer than finding the other values of a neurons neighbors.
                    for connection_pair in neuron.connections_back:
                        temp_value += connection_pair[0].value * connection_pair[1].weight

                    neuron.value = self.sigmoid(temp_value + neuron.bias)

    # The back-propagate step is where our ANN actually learns.
    # It first calculates all the errors of the neurons starting from the output layer,
    # and then calculates the delta weights and delta biases
    def back_propagate(self, l, outputs):

        if self.verbose >= 2: print("\n**** Back Propagation ***")

        for i in range(len(self.neurons[-1])):
            self.neurons[-1][i].error = outputs[l][i] - self.neurons[-1][i].value
            if self.verbose >= 2: print("Error of the output neuron", q, "during output", l, "is", self.neurons[-1][q].error)

        for i, neuron_list in reversed(list(enumerate(self.neurons))):
            for neuron in neuron_list:
                if i != len(self.neurons) - 1 and i != 0:
                    temp_error = 0

                    for connection_pair in neuron.connections_forward:
                        temp_error += connection_pair[1].weight * connection_pair[0].error

                    neuron.error = temp_error
                    if self.verbose >= 2: print("Error of the hidden neuron in layer ", neuron.layer, "during output", l, "is", neuron.error)

        for i, neuron_list in reversed(list(enumerate(self.neurons))):
            for neuron in neuron_list:
                for connection_pair in neuron.connections_forward:

                    gradient = self.sigmoid_derivative(connection_pair[0].value)
                    error_forward = connection_pair[0].error

                    adjustment_bias = ((error_forward * gradient) * self.learning_rate)
                    adjustment_weight = (((error_forward * gradient) * neuron.value) * self.learning_rate)

                    if self.verbose >= 2: print("Adjustments for weight in between layer ", i, "and", i - 1, "is going to be", adjustment_weight)
                    if self.verbose >= 2: print("Adjustments for biases in layer ", i, "is going to be", adjustment_bias)

                    connection_pair[1].weight += adjustment_weight
                    neuron.bias += adjustment_bias

                    if self.verbose >= 2: print("New for weight in between layer ", t, "and", t - 1, "is", connection_pair[1].weight)
                    if self.verbose >= 2: print("New for bias in layer ", t, "is", neuron.bias)

    def train(self, inputs, desired_output, epochs):

        self.input_output_checks(desired_output, inputs)

        for epoch in range(epochs):
            if self.verbose >= 1: print("\n", "Epoch", epoch + 1)

            for l in range(len(inputs)):

                self.feed_forward(l, inputs)

                output_neuron_outputs = []
                for m in self.neurons[-1]:
                    output_neuron_outputs.append(m.value)
                if self.verbose >= 1: print("For input: ", inputs[l], ", The output was Actual: ", desired_output[l], " – Guess: ", output_neuron_outputs, sep="")
                self.back_propagate(l, desired_output)

                output_neuron_error = []
                for n in self.neurons[-1]:
                    output_neuron_error.append(float(n.error))

                if self.verbose >= 1: print("Input: ", l + 1, " – error on output: ", output_neuron_error, sep="")

                accuracy = 0
                for o_neuron in self.neurons[-1]:
                    accuracy += o_neuron.error ** 2

                print("\r", "Epoch ", epoch + 1, " Error Mitigation: ", f"{100 - (accuracy * 100):.2f}", "%", end="", sep="")

    # This is a simple function that can flush the neurons and raise and error for the user to understand.
    def shut_down_model(self, message):
        self.neurons.clear()
        raise ValueError(message)

    # This function runs checks to see if the input and output that the user defined works for the given model
    def input_output_checks(self, check_outputs, check_inputs):
        for i, input_iter in enumerate(check_inputs):
            if len(input_iter) > len(self.neurons[0]) or len(input_iter) < len(self.neurons[0]):
                self.shut_down_model("Number of inputs does not match the number of input neurons at point " + str(i))
            for input_num in input_iter:
                if input_num > 1 or input_num < -1:
                    self.shut_down_model("Input has a value outside range [-1, 1]")

        for i, output_iter in enumerate(check_outputs):
            # TODO: Fix below, non-functional
            if len(output_iter) > len(self.neurons[-1]) or len(output_iter) < len(self.neurons[-1]):
                self.shut_down_model("Number of outputs does not match the number of output neurons")

            for j, outputs in enumerate(check_outputs):
                for output_num in outputs:
                    if output_num > 1 or output_num < -1:
                        self.shut_down_model("Output has a value outside range [-1, 1] at point " + str(j))

    # This function will feed-forward a set of inputs and return the output neurons values without back-propagation
    # Hence, a prediction.
    def predict(self, inputs):

        self.feed_forward(0, inputs)
        output_neuron_outputs = []
        for output_neuron in self.neurons[-1]:
            output_neuron_outputs.append(output_neuron.value)
        return output_neuron_outputs

    # Below are all of the activation functions we can use on our neurons values. These are chosen by the user.
    @staticmethod
    def relu(x):
        return x * (x > 0)

    @staticmethod
    def relu_derivative(x):
        return 1. * (x > 0)

    @staticmethod
    def sigmoid(x):
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)


# ************************************ Universal Neuron ****************************************
# Simply an object that can contain the info of a neuron. (Connections, value, bias, and error)
# **********************************************************************************************
class Neuron:
    def __init__(self, layer, neurons):
        self.connections_forward = []
        self.connections_back = []
        self.layer_num = layer
        self.bias = 0
        self.value = 0
        self.error = 0

        neurons[layer].append(self)

    def add_forward_connection(self, destination_neuron, connection):
        self.connections_forward.append([destination_neuron, connection])

    def add_backward_connection(self, source_neuron, connection):
        self.connections_back.append([source_neuron, connection])


# **************************** Universal Connection ********************************
# Simply an object that can contain the info of a connection. (Weight, connections)
# **********************************************************************************
class Connection:
    def __init__(self, source_neuron, destination_neuron):

        self.weight = (2 * np.random.random(1) - 1)[0]
        source_neuron.add_forward_connection(destination_neuron, self)
        destination_neuron.add_backward_connection(source_neuron, self)


# This function preps the input data to keep it between 0 and 1
def prep_array(array, divisor):
    for i in range(len(array)):
        array[i] = array[i] / divisor
    return array


# Load a saved model from a serialized npy file
def load_model(file):
    file_extension = re.search("[^.]*$", file)
    if file_extension.group(0) == "npy":
        try:
            infile = open(file, 'rb')
        except FileNotFoundError:
            raise ValueError("There is no file under the name " + file)
    else:
        try:
            infile = open(file + ".npy", 'rb')
        except FileNotFoundError:
            raise ValueError("There is no file under the name " + file + ".pickle")
    loaded_model = np.load(infile, allow_pickle=True)
    infile.close()
    return loaded_model[()]


# Save a  model into a serialized npy file
def save_model(model, file):
    file_extension = re.search("[^.]*$", file)

    if file_extension.group(0) == "npy":
        outfile = open(file, 'wb')
    else:
        outfile = open(file + ".npy", 'wb')
    np.save(outfile, model)
    outfile.close()


def tokenize(array):
    pass

    temp_array = []
    for i in range(len(array)):
        temp_array.append(i * len(array))

    for i in range(len(array)):
        temp_array[i] /= max(temp_array)

    return temp_array


def test_train_split(data, split):
    if isinstance(data, list):
        return data[:split], data[split:]
    if isinstance(data, pd.DataFrame):
        return data.iloc[:split], data.iloc[split:]

