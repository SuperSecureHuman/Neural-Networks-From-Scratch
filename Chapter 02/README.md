## Chapter 02 - Creating a neuron

---

You will require python, numpy and matplotlib for the rest of this journey.

&nbsp;

### Single neuron

---

![](./assets/neuron_3_input.png)

Here you can see that the neuron has 3 inputs (plus there is a bias)

Now, seeing this programatically, we can create a neuron with 3 inputs and a bias.

```python
inputs = [1, 2, 3]
weights = [0.2, 0.8, -0.5]
bias = 2

output = (
(inputs[0] * weights[0]) + 
(inputs[1] * weights[1]) + 
(inputs[2] * weights[2]) + bias )

print(output)

>>> 2.3
```

[Single neuron with 3 inputs example](https://nnfs.io/bkr/)

&nbsp;

Now with 4 inputs it will look like this:

```python
inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1]
bias = 2
output = (
(inputs[0] * weights[0]) +
(inputs[1] * weights[1]) +
(inputs[2] * weights[2]) +
(inputs[3] * weights[3]) + bias )

print(output)

>>> 4.8
```

### Layer of neurons

Upto now we have looked at only one neuron.

Lets step it up to a layer of neurons.

![](assets/4-3_neural_network.png)

Since we have a layer, instead of single neuron, we will now have a set of weights and biases.

Number of wights + biases pairs will be equal to the number of neurons in the output layer.

```python
inputs = [1, 2, 3, 2.5]
weights1 = [ 0.2,    0.8,   -0.5,    1]
weights2 = [ 0.5,   -0.91,   0.26,  -0.5]
weights3 = [-0.26,  -0.27,   0.17,   0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

outputs = [
    #Neuron 1
    (inputs[0] * weights1[0]) +
    (inputs[1] * weights1[1]) +
    (inputs[2] * weights1[2]) +
    (inputs[3] * weights1[3]) + bias1

    #Neuron 2
    (inputs[0] * weights2[0]) +
    (inputs[1] * weights2[1]) +
    (inputs[2] * weights2[2]) +
    (inputs[3] * weights2[3]) + bias2

    #Neuron 3

    (inputs[0] * weights3[0]) +
    (inputs[1] * weights3[1]) +
    (inputs[2] * weights3[2]) +
    (inputs[3] * weights3[3]) + bias3
]

print(outputs)

>>>[4.8,1.21, 2.385]

```

Visulaization of this code - [3 neuron layer with 4 inputs](https://nnfs.io/mxo/)

&nbsp;

### Making the same, but with loops in python

```python

inputs = [1, 2, 3, 2.5]

weights = [ [0.2, 0.8, -0.5, 1],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87] ]

biases = [2, 3, 0.5]

# Output of current layer
layer_outputs = []

# For each neuron
for neuron_weights, neuron_bias in zip(weights, biases):

# Zeroed output of given neuron
neuron_output = 0

# For each input and weight to the neuron
for n_input, weight in zip(inputs, neuron_weights):
# Multiply this input by associated weight
# and add to the neuron???s output variable

neuron_output += n_input*weight
# Add bias

neuron_output += neuron_bias
# Put neuron???s result to the layer???s output list

layer_outputs.append(neuron_output)

print(layer_outputs)


>>> [4.8, 1.21, 2.385]
```

Note - If you dont know about the zip() used in the above python snippet, refer here - https://realpython.com/python-zip-function/

### About matrix operations

&nbsp;
You should have a understanding of the following:



* Dot product
* Vector Addition
* Matrix Multiplication
* Transpose of a matrix

You can refer to khan academy for enough understanding on these topics.

I did not document it here, because I have already know it :sweat_smile: 

### Switching gears to numpy!

Numpy is da facto python library for matrix operations.

Now, we will replicate the above codes, but in numpy instead of pure python.

Numpy not only makes it easier to code, but also makes processing faster.

You are not likely to notice much of a performance difference in the code mentioned in this chapter, but when it 
comes to large matrices, the difference stars to be noticeable.

&nbsp;

#### Single neuron with numpy


```python
import numpy as np

inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1]
bias = 2

#In the above codes, we did something like element wise multiplication. That is precisely what dot product does.
outputs = np.dot(inputs, weights) + bias 

print(outputs)

>>> 4.8
```

#### Layer of neurons with numpy

```python
import numpy as np
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [ [0.2, 0.8, -0.5, 1],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]
layer_outputs = np.dot(weights, inputs) + biases

print(layer_outputs)
>>> array([4.8, 1.21, 2.385])
```


### Batch of Data

You can think of batches as a set of inputs, from the entire dataset.

The reason we need to care about batches is that we use it prallelize our computations.

And another reason is that we dont want to over fit the model. This happens when the network learn the training data too perfectly, and makes correct predictions on train data, but fails to generalize to new data.

The size of batch you select is kinda important. Look at the following animation to know why.

[How batches can help with fitment](https://nnfs.io/vyu/)

In this case, 'fit' means you drawing a line that best fits the data. Method of least squares is used to find that line.

A batch will look something like this

```python
batch = [
    [array1],
    [array2],
    [array3],
    [array4],
    [array5],
    [array6]
]

#This is example with batch size being 6 
```

### Layer of neurons in numpy with batches of data

Note - Here, I assume you have idea of dot product and transpose of a matrix.

Here, you need to transpose the weights matrix, and then do dot product.
The reason to do transpose can be seen here - [Why we need to transpose weights](https://nnfs.io/crq/)

```python
import numpy as np
inputs = [  [1.0, 2.0, 3.0, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8]]

weights = [ [0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]
layer_outputs = np.dot(inputs, np.array(weights).T) + biases
print(layer_outputs)
>>>
array([[4.8, 1.21, 2.385],
[8.9, -8.91, 0.2],
[1.41, 1.051, 0.026]])
```

The code visualized:

[Matrix product with row and column vectors with a batch of inputs to the neural network](https://nnfs.io/gjw/)

[Adding biases after the matrix product from a batch of inputs](https://nnfs.io/qty/)

---
Chapter 2 of nnfs book