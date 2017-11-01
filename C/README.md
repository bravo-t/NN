# A neural network framework written in C
### Table of contents
1. [Fully connected network](#fully-connected-network)
2. [Convolutional neural network](#convolutional-neural-network)
## Fully connected network
* What is this?
  * This is a fully connected neural network engine I wrote to figure out how a neural network works.
  * The activation function used in this network is leaky ReLU, and the classifier is softmax.
  * It uses SGD as the default update strategy, and it also has momentum update, NAG update and RMSProp built in.
  * In "train" mode, this program takes in a configuration file, trians itself, and dumps out the learned weights.
  * In "test" mode, this program reads in the weights dumped out earlier, and calculate scores on the input date set.
* How to compile
  * Clone, or download all the source files.
  * Compile with `make`.
* How to use
  * You have to create a config file for network configuration. An example of it is [fc.config](fc.config).
    * The syntax of this config file is like `key = value`.
    * `data_set` is the ASCII file that contains input dataset for training or testing
    * `correct_labels` is the ASCII file that contains correct labels for the training dataset.
    * `hidden_layer_sizes` is the parameter that defined how the network looks like. 
      * For example `hidden_layer_sizes = 100,3` means this network will have 2 hidden layers, and first hidden layer will have a width of 100, and the second hidden layer is the score layer, its width is defined by the number of labels this network will calculate.
      * Number of hidden layers is determined by how many sizes you give it.
      * You don't have control on the last layer of the network.
    * `alpha` is the hyperparameter used in leaky ReLU. If you set it to 0, the activation function becomes ReLU.
    * `labels` is how many labels this network is going to calculate for each training example.
    * `params_dir` specifies the directory that the network dumps and loads learned weights.
    * The rest of the parameters are quite self-explanatory.
  * You will also need to create the file contains dataset and correct labels.
    * Data structures used in this network are represented by 2-D matrixes, so the input data are 2-D matrixes also.
    * Example for the input data can be find at [test_data/correct_labels.txt](test_data/correct_labels.txt).
    * The format is described as following:
      * The first line contains 3 parts: a name (pick one whatever you like, cannot has spaces in it), height of the dataset, width of the dataset.
      * The rest of the lines are just data.
      * Correct labels are matrixes with a width of 1, and a same height as the dataset, represents the correct index of the scores, which contains `labels` elements.
      * Below is an example:
```
      W 3 6
      w11 w12 w13 w14 w15 w16
      w21 w22 w23 w24 w25 w26
      w31 w32 w33 w34 w35 w36
```
* Example
  * Below is an example I created based on the [small case of CS213n](http://cs231n.github.io/neural-networks-case-study/).
  * Network configuration file is [fc.config](fc.config).
  * The input dataset is [test_data/training_data.txt](test_data/training_data.txt).
  * The correct labels to train the network can be found at [test_data/correct_labels.txt](test_data/correct_labels.txt).
  * Below is the messages while running it:
```
    $ ./fully_connected_network fc.config
    INFO: This network consists of 2 hidden layers, and their sizes are configured to be 100 3 
    INFO: Initializing all required learnable parameters for the network
    INFO: 2 W matrixes, 500 learnable weights initialized, 4.00 KB meomry used
    INFO: 2 b matrixes, 103 learnable biases initialized, 0.90 KB meomry used
    INFO: 2 H matrixes, 30900 learnable hidden layer values initialized, 241.50 KB meomry used
    INFO: A total number of 246.40 KB memory is used by learnable parameters in the network
    INFO: Training network
    INFO: Epoch 1000, data loss: 0.104390, regulization loss: 4.053989, total loss: 4.158380
    INFO: Epoch 2000, data loss: 0.322864, regulization loss: 5.449154, total loss: 5.772018
    INFO: Epoch 3000, data loss: 0.239057, regulization loss: 5.338373, total loss: 5.577430
    INFO: Epoch 4000, data loss: 0.037477, regulization loss: 6.129504, total loss: 6.166981
    INFO: Epoch 5000, data loss: 0.661693, regulization loss: 5.666165, total loss: 6.327857
    INFO: Epoch 6000, data loss: 0.054928, regulization loss: 6.936616, total loss: 6.991545
    INFO: 97.000000% correct on training data
    INFO: Network parameters dumped to ./network.params
```
  * And as specified, you will get all learned weights dumped out in [network.params](network.params), which can be loaded later for testing

## Convolutional neural network
* What is this?
  
* How to compile
  
* How to use
  
* Example

