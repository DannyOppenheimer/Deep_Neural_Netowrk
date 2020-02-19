import numpy as np
import sys


# The main class that handles the creation of the ANN model, training (including feeding forward and back propagation),
# math like sigmoid and the derivative of sigmoid, and predicting unknown values.
# TODO: Make new classes for RNNs and CNNs (a big task)
class ANN:

    def __init__(self, layers, verbose=0, learning_rate=0.1):
        # Setting a random seed is a good idea,
        # so we can get reproducible results and tweak our layer sizes for experimentation
        np.random.seed(1)

        self.layers = len(layers)
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.neurons = []

        self.loading_signs = ["[|]", "[/]", "[–]", "[\\]"]

        loop_num = 0
        percentage_total = 0
        percentage_current = 0
        array_lang = " ".join(("Creating array with", str(len(layers)), "layers:"))
        neuron_lang = "Adding neurons to array."
        connection_lang = "Adding connections to each neuron"

        # Create an array with the specified layer size
        for _ in layers:
            loop_num += 1
            print("\r", " ".join((array_lang, self.loading_signs[loop_num % 4])), end="")
            # Each of these arrays-in-arrays will hold a layers worth of neurons
            self.neurons.append([])

        print("\r", " ".join((array_lang, "[✓]")), end="")
        print("\n")
        # Add the number of specified neurons to each layer
        for i in range(len(layers)):
            for _ in range(layers[i]):
                loop_num += 1
                percentage_total += 1
                print("\r", " ".join((neuron_lang, self.loading_signs[loop_num % 4])), end="")
                Neuron(i, self.neurons)
        print("\r", " ".join((neuron_lang, "[✓]")), end="")
        print("\n")

        # Link all neurons with connections
        for i in range(len(self.neurons)):
            for neuron in self.neurons[i]:
                percentage_current += 1
                print("\r", " ".join((connection_lang, str(int(((percentage_current / percentage_total) * 100))), "%",
                                      self.loading_signs[percentage_current % 4])), end="", sep="")

                if i != len(self.neurons) - 1:
                    for forward_neuron in self.neurons[i + 1]:
                        Connection(neuron, forward_neuron)
        print("\r", " ".join((connection_lang, "[✓]")), end="")
        print("\n")

    def feed_forward(self, l, inputs):

        # This is our feed-forward step. We are first initializing our input neurons with the input values
        for i in range(len(self.neurons[0])):
            self.neurons[0][i].value = inputs[l][i]

        # Now we set all the other neurons values
        for i in range(len(self.neurons)):
            if i != 0:
                for neuron in self.neurons[i]:
                    # Create a temporary value to add our Weight * NeuronVal to
                    temp_value = 0

                    # For each connection backwards, set neuron"s own value to (Weight * NeuronVal) + Bias
                    for connection_pair in neuron.connections_back:
                        temp_value += connection_pair[0].value * connection_pair[1].weight

                    neuron.value = self.sigmoid(temp_value + neuron.bias)

    def back_propagate(self, l, outputs):

        # For a lot of these, we will do a reversed enumerated list in order to back propagate, well, backwards

        if self.verbose >= 2: print("\n**** Back Propagation ***")

        # Now, we can calculate the error of the output layers" neurons
        for i in range(len(self.neurons[-1])):
            self.neurons[-1][i].error = outputs[l][i] - self.neurons[-1][i].value
            if self.verbose >= 2: print("Error of the output neuron", q, "during output", l, "is", self.neurons[-1][q].error)

        # Here we calculate the error of all the hidden layers" neurons
        for i, neuron_list in reversed(list(enumerate(self.neurons))):
            for neuron in neuron_list:
                if i != len(self.neurons) - 1 and i != 0:
                    # The value to store the total change as we iterate through all the errors
                    temp_error = 0

                    for connection_pair in neuron.connections_forward:
                        temp_error += connection_pair[1].weight * connection_pair[0].error

                    neuron.error = temp_error
                    if self.verbose >= 2: print("Error of the hidden neuron in layer ",neuron.layer, "during output", l, "is", neuron.error)

        # Finally, we calculate and set the weights accordingly
        for i, neuron_list in reversed(list(enumerate(self.neurons))):
            for neuron in neuron_list:
                for connection_pair in neuron.connections_forward:

                    # This is the gradient we want to descend upon to minimise the error of our outputs
                    gradient = self.sigmoid_derivative(connection_pair[0].value)
                    # This is the error of the neuron that the weight affects forward
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

        # Real training time!
        for epoch in range(epochs):
            if self.verbose >= 1: print("\n", "Epoch", epoch + 1)

            # For each input, run this loop!
            for l in range(len(inputs)):

                self.feed_forward(l, inputs)

                # get all outputs
                output_neuron_outputs = []
                for m in self.neurons[-1]:
                    output_neuron_outputs.append(m.value)
                if self.verbose >= 1: print("For input: ", inputs[l], ", The output was Actual: ", desired_output[l], " – Guess: ", output_neuron_outputs, sep="")
                self.back_propagate(l, desired_output)

                # get all output errors
                output_neuron_error = []
                for n in self.neurons[-1]:
                    output_neuron_error.append(float(n.error))

                if self.verbose >= 1: print("Input: ", l + 1, " – error on output: ", output_neuron_error, sep="")

            accuracy = 0
            for o_neuron in self.neurons[-1]:
                accuracy += (o_neuron.error) ** 2

            if self.verbose == 0: print("\r", "Epoch: ", epoch + 1, " Accuracy: ", f"{100 - (accuracy * 100):.2f}", "%", end="", sep="")

    @staticmethod
    def sigmoid(x):
        # The sigmoid function is an S shaped line that will squash our numbers down in between 0 and 1,
        # sort of like the probability of that neuron being the one we want.
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    def shut_down_model(self, message):
        self.neurons.clear()
        sys.exit(message)

    def input_output_checks(self, check_outputs, check_inputs):
        # Inputs and outputs will come in as [[X, X], [X, X]]
        # Check if inputs match input layer, and if any are above 1 or below -1
        for input_iter in check_inputs:
            if len(input_iter) > len(self.neurons[0]) or len(input_iter) < len(self.neurons[0]):
                self.shut_down_model("Number of inputs does not match the number of input neurons")
            for i, input_num in enumerate(input_iter):
                if input_num > 1 or input_num < -1:
                    self.shut_down_model("Detected input with a value outside range [-1, 1]")
                try:
                    check_outputs[i]
                except IndexError:
                    self.shut_down_model("The inputs do not match the outputs!")

        # Check if outputs match ouput layer, and if any are above 1 or below -1
        for i, output_iter in enumerate(check_outputs):
            # TODO: Fix below, non-functional
            if len(output_iter) > len(self.neurons[-1]) or len(output_iter) < len(self.neurons[-1]):
                self.shut_down_model("Number of outputs does not match the number of output neurons")

            for j, output_num in enumerate(input_iter):
                if output_num > 1 or output_num < -1:
                    self.shut_down_model("Detected output with a value outside range [-1, 1]")
                try:
                    check_inputs[j]
                except IndexError:
                    self.shut_down_model("The outputs do not match the inputs!")

    def predict(self, inputs):

        self.feed_forward(0, inputs)

        output_neuron_outputs = []
        for output_neuron in self.neurons[-1]:
            output_neuron_outputs.append(output_neuron.value)
        return output_neuron_outputs


class Neuron:
    def __init__(self, layer, neurons):
        self.connections_forward = []
        self.connections_back = []
        self.layer_num = layer

        self.value = 0
        self.bias = 0
        self.error = 0

        neurons[layer].append(self)

    def add_forward_connection(self, destination_neuron, connection):
        self.connections_forward.append([destination_neuron, connection])

    def add_backward_connection(self, source_neuron, connection):
        self.connections_back.append([source_neuron, connection])


class Connection:
    def __init__(self, source_neuron, destination_neuron):

        self.weight = (np.random.random(1) - 1)[0]
        source_neuron.add_forward_connection(destination_neuron, self)
        destination_neuron.add_backward_connection(source_neuron, self)


def prep_array(array, divisor):
    for i in range(len(array)):
        array[i] = array[i] / divisor
    return array
