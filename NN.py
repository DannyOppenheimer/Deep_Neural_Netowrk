import numpy as np
import sys
from PIL import Image as im

neurons = []
loading_signs = ['[|]', '[/]', '[–]', '[\\]']


class Brain:

    def __init__(self, layers):

        self.layers = len(layers)
        self.learning_rate = 0.1
        self.cost = 0

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
                                      loading_signs[percentage_current % 4])), end='', sep='')
                if m != len(neurons) - 1:
                    for o in range(len(neurons[m + 1])):
                        Connection(neurons[m][n], neurons[m + 1][o])

        # TODO: Make sure that all connections are in the right spots for back propagation

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

    def back_propagate(self, l, outputs, verbose=False):
        # for a lot of these, we will do a reversed enumerated list in order to back propagate, well, backwards

        if verbose: print('\n**** Back Propogatation ***')

        # now, we can calculate the error of the output
        for q in range(len(neurons[len(neurons) - 1])):
            neurons[len(neurons) - 1][q].error = float(outputs[l][q]) - float(neurons[len(neurons) - 1][q].value)
            if verbose: print('Error of the output neuron', q, 'during output', l, 'is', neurons[len(neurons) - 1][q].error)

        # Here we calculate the error of all the hidden layers' neurons
        for r, s in reversed(list(enumerate(neurons))):
            for x in range(len(neurons[r])):
                if r != len(neurons) - 1 and r != 0:
                    temp_error = 0

                    for z in range(len(neurons[r][x].connections_forward)):
                        temp_error += neurons[r][x].connections_forward[z][1].weight * neurons[r][x].connections_forward[z][0].error

                    neurons[r][x].error = float(temp_error)
                    if verbose: print('Error of the hidden neuron', x, ' in layer ', r, 'during output', l, 'is', neurons[r][x].error)

        # Finally, we calculate and set the weights accordingly
        for t, u in reversed(list(enumerate(neurons))):
            for v in range(len(neurons[t])):
                for w in range(len(neurons[t][v].connections_forward)):

                    gradient = self.sigmoid_derivative(neurons[t][v].connections_forward[w][0].value)
                    error_forward = neurons[t][v].connections_forward[w][0].error

                    adjustment_bias = ((error_forward * gradient) * self.learning_rate)
                    adjustment_weight = (((error_forward * gradient) * neurons[t][v].value) * self.learning_rate)

                    if verbose: print('Adjustments for weight in between layer ', t, 'and', t - 1, 'is going to be', adjustment_weight)
                    if verbose: print('Adjustments for biases in layer ', t, 'is going to be', adjustment_bias)

                    neurons[t][v].connections_forward[w][1].weight += float(adjustment_weight)
                    neurons[t][v].bias = float(adjustment_bias) + float(neurons[t][v].bias)

                    if verbose: print('New for weight in between layer ', t, 'and', t - 1, 'is', neurons[t][v].connections_forward[w][1].weight)
                    if verbose: print('New for bias in layer ', t, 'is', neurons[t][v].bias)

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
            print('\n', "Epoch", k)

            # For each input, run this loop!
            for l in range(len(inputs)):

                self.feed_forward(l, inputs)

                # get all outputs
                output_neuron_outputs = []
                for m in neurons[len(neurons) - 1]:
                    output_neuron_outputs.append(m.value)
                print("For input: ", inputs[l], ", The output was Actual: ", desired_output[l], " – Guess: ", output_neuron_outputs, sep='')
                self.back_propagate(l, desired_output, verbose=False)

                # get all output errors
                output_neuron_error = []
                for n in neurons[len(neurons) - 1]:
                    output_neuron_error.append(float(n.error))

                print("Input: ", l + 1, " – error on output: ", output_neuron_error, sep='')

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
        for i in range(len(inputs)):
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


def load_images():

    white1 = np.array(im.open('Images/white1.jpg', 'r').convert('RGB').getdata()).flatten().tolist()
    white1 = prep_array(white1, 255)
    white2 = np.array(im.open('Images/white2.jpg', 'r').convert('RGB').getdata()).flatten().tolist()
    white2 = prep_array(white2, 255)
    white3 = np.array(im.open('Images/white3.jpg', 'r').convert('RGB').getdata()).flatten().tolist()
    white3 = prep_array(white3, 255)
    white4 = np.array(im.open('Images/white4.jpg', 'r').convert('RGB').getdata()).flatten().tolist()
    white4 = prep_array(white4, 255)
    white5 = np.array(im.open('Images/white5.jpg', 'r').convert('RGB').getdata()).flatten().tolist()
    white5 = prep_array(white5, 255)
    black1 = np.array(im.open('Images/black1.jpg', 'r').convert('RGB').getdata()).flatten().tolist()
    black1 = prep_array(black1, 255)
    black2 = np.array(im.open('Images/black2.jpg', 'r').convert('RGB').getdata()).flatten().tolist()
    black2 = prep_array(black2, 255)
    black3 = np.array(im.open('Images/black3.jpg', 'r').convert('RGB').getdata()).flatten().tolist()
    black3 = prep_array(black3, 255)
    black4 = np.array(im.open('Images/black4.jpg', 'r').convert('RGB').getdata()).flatten().tolist()
    black4 = prep_array(black4, 255)
    black5 = np.array(im.open('Images/black5.jpg', 'r').convert('RGB').getdata()).flatten().tolist()
    black5 = prep_array(black5, 255)

    return [white1, white2, white3, white4, white5, black1, black2, black3, black4, black5]


def prep_array(array, divisor):
    for i in range(len(array)):
        array[i] = array[i] / divisor
    return array


if __name__ == "__main__":

    # Setting a random seed is a good idea,
    # so we can get reproducible results and tweak our layer sizes for experimentation
    np.random.seed(1)

    brain = Brain([2, 3, 2])

    brain.train([[0, 1], [1, 0]],
                # TODO: implement multiple neuron output layer
                [[0, 1], [1, 0]],

                10000)

