import numpy as np
import sys


neurons = []
loading_signs = ['[|]', '[/]', '[–]', '[\\]']


class Brain:
    def __init__(self, layers):

        self.layers = len(layers)
        self.learning_rate = 0.1

        loop_num = 0
        percentage_total = 0
        percentage_current = 0
        array_lang = " ".join(("Creating array with", str(len(layers)), "layers:"))
        neuron_lang = "Adding neurons to array."
        connection_lang = "Adding connections to each neuron"

        # Create an array with the specified layer size
        for j in range(len(layers)):

            loop_num += 1
            print('\r', " ".join((array_lang, loading_signs[loop_num % 4])), end='')

            neurons.append([])

        print('\n')
        # Add the number of specified neurons to each layer
        for k in range(len(layers)):
            for l in range(layers[k]):
                loop_num += 1
                percentage_total += 1
                print('\r', " ".join((neuron_lang, loading_signs[loop_num % 4])), end='')
                Neuron(k)

        print('\n')

        # Link all neurons with connections
        for m in range(len(neurons)):
            for n in range(len(neurons[m])):
                percentage_current += 1
                print('\r', " ".join((connection_lang, str(int(((percentage_current / percentage_total) * 100))), '%',
                                      loading_signs[percentage_current % 4])), end='')
                if m != len(neurons) - 1:
                    for o in range(len(neurons[m + 1])):
                        Connection(neurons[m][n], neurons[m + 1][o])

        # TODO: Make sure that all connections are in the right spots for back propagation

        print('\n')

    def feed_forward(self, l, inputs):
        print("Run number:", l)

        # This is our feed-forward step. We are first initializing our input neurons with the input values
        for m in range(len(neurons[0])):
            neurons[0][m].value = inputs[l][m]
        # Now we set all the other neurons values
        for n in range(len(neurons)):
            for o in range(len(neurons[n])):
                # create a temporary value to add our (W + NVal) + B to
                temp_value = 0

                # for each connection backwards, set neuron's own value to (W + NVal) + B
                for p in range(len(neurons[n][o].connections_back)):
                    temp_value += \
                        (neurons[n][o].connections_back[p][0].value * neurons[n][o].connections_back[p][1].weight) \
                        + neurons[n][o].bias
                    neurons[n][o].value = self.sigmoid(temp_value)

    def back_propagate(self, l, outputs):

        # now, we can calculate the error of the output
        for q in range(len(neurons[len(neurons) - 1])):
            neurons[len(neurons) - 1][q].error = outputs[l] - neurons[len(neurons) - 1][q].value

        # Here we calculate the error of all the hidden layers' neurons
        for r, s in reversed(list(enumerate(neurons))):
            for x in range(len(neurons[r])):
                if r != len(neurons) - 1 and r != 0:
                    temp_error = 0
                    for y in range(len(neurons[r][x].connections_forward)):
                        temp_error += neurons[r][x].connections_forward[y][1].weight * neurons[r][x].connections_forward[y][0].error
                    neurons[r][x].error += temp_error

        # Finally, we calculate and set the weights accordingly
        for t, u in reversed(list(enumerate(neurons))):
            for v in range(len(neurons[t])):
                for w in range(len(neurons[t][v].connections_forward)):
                    adjustment_bias = self.learning_rate \
                                       * (neurons[t][v].connections_forward[w][0].error * self.sigmoid_derivative(neurons[t][v].connections_forward[w][0].value))

                    adjustment_weight = self.learning_rate\
                                         * ((neurons[t][v].connections_forward[w][0].error * self.sigmoid_derivative(neurons[t][v].connections_forward[w][0].value))
                                            * neurons[t][v].value)

                    neurons[t][v].connections_forward[w][1].weight += adjustment_weight
                    neurons[t][v].bias += adjustment_bias

    def train(self, inputs, outputs, epochs):

        # Check if inputs match input layer, and if any are above 1 or below -1
        for i in range(len(inputs)):
            if len(inputs[i]) > len(neurons[0]) or len(inputs[i]) < len(neurons[0]):
                self.shut_down_model("Number of inputs does not match the number of input neurons")
            for j in range(len(inputs[i])):
                if inputs[i][j] > 1 or inputs[i][j] < -1:
                    self.shut_down_model("Detected input with a value outside range [-1, 1]")

        # Real training time!
        for k in range(epochs):
            print('\n', "Epoch", k)
            # For each input, run this loop!
            for l in range(len(inputs)):

                self.feed_forward(l, inputs)
                print("Actual", inputs[l], ": Guess:", neurons[len(neurons) - 1][0].value)
                self.back_propagate(l, outputs)
                print("Input", l, ": error on output:", neurons[len(neurons) - 1][0].error)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    @staticmethod
    def calc_cost(desired_output, actual_output):
        return (actual_output - desired_output) ** 2

    @staticmethod
    def shut_down_model(message):
        sys.exit(message)


class Neuron:
    def __init__(self, layer):
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
        self.weight = (2 * np.random.random(1) - 1)[0]

        source_neuron.add_forward_connection(destination_neuron, self)
        destination_neuron.add_backward_connection(source_neuron, self)

    def set_weight(self, x):
        self.weight = x


if __name__ == "__main__":

    # Setting a random seed is a good idea,
    # so we can get reproducible results and tweak our layer sizes for experimentation
    np.random.seed(2)

    brain = Brain([2, 2, 1])

    brain.train([[1, 1], [0, 0]],

                [1, 0],

                5000)
