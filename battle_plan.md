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
1. Add up squares of the differences of what you got vs what you want for cost (received - desired)^2

#### **Calculate error**:
1. (known answer - system guess)

#### **Back Propagate**:
1. Learn calculus

weights
output of the hidden layer passed through the derivative of sigmoid
multiply above by the output errors of the neurons
multiply above my the learning rate
multiply above by the input of the last neuron

biases
output of the hidden layer passed through the derivative of sigmoid
multiply above by the output errors of the neurons
multiply above my the learning rate


I -  H \
  \/
  /\     O
I -  H /

input to first hidden:
learning rate * hidden error * derivative(hidden) * input
