# Neural Network Battle Plan

## Have layers of neurons as neuron objects in an array

**Index 0**: IN IN IN IN IN IN

**Index 1**: - H H H H H

**Index 2**: -- H H H H

**Index 3**: ---- H H

**Index 4**: ------ O

For each of these layers, create a connection object forwards and backwards to respective neurons (meaning its all connected)

Steps:

#### **Feed Forward**:
1. Check to make sure the input works for the model
2. Then, set the input neurons with their input values
3. For every other neuron starting at layer 1, set their value to the values from all connected neurons, multiplied by all connected weights, and added to its bias. (All this is squished by a sigmoid function)
4. Stop at output

#### **Calculate cost**:
(system guess - known answer)^2

#### **Calculate error**:
(known answer - system guess)

#### **Back Propagate**:
1. FOR WEIGHTS:
2. output of the hidden layer passed through the derivative of sigmoid
3. multiply above by the output errors of the neurons
4. multiply above my the learning rate
5. multiply above by the input of the last neuron
>
**learning rate * hidden error * derivative(hidden) * input**
>
1. FOR BIASES
2. output of the hidden layer passed through the derivative of sigmoid
3. multiply above by the output errors of the neurons
4. multiply above my the learning rate
>
**learning rate * hidden error * derivative(hidden)**
