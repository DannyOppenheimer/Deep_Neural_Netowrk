import numpy as np
import sys

neurons = []
loading_signs = ['[|]', '[/]', '[–]', '[\\]']


# The main class that handles the creation of the model, training (this includes feeding forward and back propagation),
# math like sigmoid and the derivative of sigmoid, and predicting unknown values.
class Brain:

    def __init__(self, layers, verbose=0):
        # Setting a random seed is a good idea,
        # so we can get reproducible results and tweak our layer sizes for experimentation
        np.random.seed(1)

        self.layers = len(layers)
        self.learning_rate = 0.1
        self.cost = 0
        self.verbose = verbose

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
        print('\r', " ".join((array_lang, "[✓]")), end='')
        print('\n')
        # Add the number of specified neurons to each layer
        for k in range(len(layers)):
            for l in range(layers[k]):
                loop_num += 1
                percentage_total += 1
                print('\r', " ".join((neuron_lang, loading_signs[loop_num % 4])), end='')
                Neuron(k)
        print('\r', " ".join((neuron_lang, "[✓]")), end='')
        print('\n')

        # Link all neurons with connections
        for m in range(len(neurons)):
            for n in range(len(neurons[m])):
                percentage_current += 1
                print('\r', " ".join((connection_lang, str(int(((percentage_current / percentage_total) * 100))), '%',
                                      loading_signs[percentage_current % 4])), end='', sep='')
                if m != len(neurons) - 1:
                    for o in range(len(neurons[m + 1])):
                        Connection(neurons[m][n], neurons[m + 1][o])

        # TODO: Make sure that all connections are in the right spots for back propagation
        print('\r', " ".join((connection_lang, "[✓]")), end='')
        print('\n')

    def feed_forward(self, l, inputs):

        # This is our feed-forward step. We are first initializing our input neurons with the input values
        for m in range(len(neurons[0])):
            neurons[0][m].value = inputs[l][m]

        # Now we set all the other neurons values
        for n in range(len(neurons)):
            if n != 0:
                for o in range(len(neurons[n])):
                    # Create a temporary value to add our (W + NVal) + B to
                    temp_value = 0

                    # For each connection backwards, set neuron's own value to (W * NVal) + B
                    for p in range(len(neurons[n][o].connections_back)):
                        temp_value += neurons[n][o].connections_back[p][0].value * neurons[n][o].connections_back[p][1].weight

                    neurons[n][o].value = self.sigmoid(temp_value + neurons[n][o].bias)

    def back_propagate(self, l, outputs):
        # for a lot of these, we will do a reversed enumerated list in order to back propagate, well, backwards

        if self.verbose >= 2: print('\n**** Back Propogatation ***')

        # now, we can calculate the error of the output
        for q in range(len(neurons[len(neurons) - 1])):
            neurons[len(neurons) - 1][q].error = float(outputs[l][q]) - float(neurons[len(neurons) - 1][q].value)
            if self.verbose >= 2: print('Error of the output neuron', q, 'during output', l, 'is', neurons[len(neurons) - 1][q].error)

        # Here we calculate the error of all the hidden layers' neurons
        for r, s in reversed(list(enumerate(neurons))):
            for x in range(len(neurons[r])):
                if r != len(neurons) - 1 and r != 0:
                    temp_error = 0

                    for z in range(len(neurons[r][x].connections_forward)):
                        temp_error += neurons[r][x].connections_forward[z][1].weight * neurons[r][x].connections_forward[z][0].error

                    neurons[r][x].error = float(temp_error)
                    if self.verbose >= 2: print('Error of the hidden neuron', x, ' in layer ', r, 'during output', l, 'is', neurons[r][x].error)

        # Finally, we calculate and set the weights accordingly
        for t, u in reversed(list(enumerate(neurons))):
            for v in range(len(neurons[t])):
                for w in range(len(neurons[t][v].connections_forward)):

                    gradient = self.sigmoid_derivative(neurons[t][v].connections_forward[w][0].value)
                    error_forward = neurons[t][v].connections_forward[w][0].error

                    adjustment_bias = ((error_forward * gradient) * self.learning_rate)
                    adjustment_weight = (((error_forward * gradient) * neurons[t][v].value) * self.learning_rate)

                    if self.verbose >= 2: print('Adjustments for weight in between layer ', t, 'and', t - 1, 'is going to be', adjustment_weight)
                    if self.verbose >= 2: print('Adjustments for biases in layer ', t, 'is going to be', adjustment_bias)

                    neurons[t][v].connections_forward[w][1].weight += float(adjustment_weight)
                    neurons[t][v].bias = float(adjustment_bias) + float(neurons[t][v].bias)

                    if self.verbose >= 2: print('New for weight in between layer ', t, 'and', t - 1, 'is', neurons[t][v].connections_forward[w][1].weight)
                    if self.verbose >= 2: print('New for bias in layer ', t, 'is', neurons[t][v].bias)

    def train(self, inputs, desired_output, epochs):

        # Check if inputs match input layer, and if any are above 1 or below -1
        for i in range(len(inputs)):
            if len(inputs[i]) > len(neurons[0]) or len(inputs[i]) < len(neurons[0]):
                self.shut_down_model("Number of inputs does not match the number of input neurons")
            for j in range(len(inputs[i])):
                if inputs[i][j] > 1 or inputs[i][j] < -1:
                    self.shut_down_model("Detected input with a value outside range [-1, 1]")

        # Real training time!
        for k in range(epochs):
            if self.verbose >= 1: print('\n', "Epoch", k + 1)

            # For each input, run this loop!
            for l in range(len(inputs)):

                self.feed_forward(l, inputs)

                # get all outputs
                output_neuron_outputs = []
                for m in neurons[len(neurons) - 1]:
                    output_neuron_outputs.append(m.value)
                if self.verbose >= 1: print("For input: ", inputs[l], ", The output was Actual: ", desired_output[l], " – Guess: ", output_neuron_outputs, sep='')
                self.back_propagate(l, desired_output)

                # get all output errors
                output_neuron_error = []
                for n in neurons[len(neurons) - 1]:
                    output_neuron_error.append(float(n.error))

                if self.verbose >= 1: print("Input: ", l + 1, " – error on output: ", output_neuron_error, sep='')

            accuracy = 0
            for o_neuron in neurons[len(neurons) - 1]:
                accuracy += (o_neuron.error) ** 2

            if self.verbose == 0: print('\r', "Epoch: ", k + 1, " Accuracy: ", f'{100 - (accuracy * 100):.2f}', "%", end='', sep='')

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
        # return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    @staticmethod
    def shut_down_model(message):
        neurons.clear()
        sys.exit(message)

    def predict(self, inputs):

        self.feed_forward(0, inputs)

        output_neuron_outputs = []
        for j in range(len(neurons[len(neurons) - 1])):
            output_neuron_outputs.append(neurons[len(neurons) - 1][j].value)
        return output_neuron_outputs


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

        self.weight = (np.random.random(1) - 1)[0]
        source_neuron.add_forward_connection(destination_neuron, self)
        destination_neuron.add_backward_connection(source_neuron, self)


def prep_array(array, divisor):
    for i in range(len(array)):
        array[i] = array[i] / divisor
    return array
