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
      * Run it using `./fully_connected_network fc.config`.
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
  * This is the convolutional neural network engine that I wrote to take a peek at CNNs
  * The network has RMSProp built in to speed up the learning process, as vanilla update isn't going to get something meaningful in reasonable time
  * Like the fully connected network, the CNN engine also takes a configuration file, and dumps out the learned weights after several epochs in train mode
  * And it will also load the learned weights and perform test on new data
* How to compile
  * Clone, or download all the source files.
  * Compile with `make`.
* How to use
  * A configuration file is needed to tell the engine how will be network look like. The syntax is the same as the fully connected network config fule, the example can be found at [cnn.config](cnn.config).
    * `data_set` is the data that will go through training, or testing. The value points to a ASCII file like it did before, but now the file contains 3D matrices.
    * `number_of_samples` is the number of matrices you have in the file pointed by `data_set`.
    * `correct_labels` is the same meaning and the contents it points to is also the same as before.
    * `M`, `N` and `hidden_layer_sizes` are the parameters the determine how are conv layers, pooling layers and fully connected layers organized. The style was taken from CS231n, which is `INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC`, where `K` is the number of elements in `hidden_layer_sizes`. You can refer to [ConvNet Architectures part of CS231n](http://cs231n.github.io/convolutional-networks/#architectures) for detailed explanation.
    * `filter_number` is the numbers of filters in each conv layer. There should be `M*N` elements for this keyword.
      * For example in [cnn.config](cnn.config), `filter_number = 16,16` means each conv layer will have 16 filters.
    * `filter_stride` is the stride for filters in each conv layer. There should also be `M*N` elements for it.
      * I know it may look strange as nobody did this, but you can actually specify different stride for X and Y direction with `filter_stride_x` and `filter_stride_y`
    * `filter_size` is the sizes the filters in each conv layer. Again, `M*N` elements for it.
      * You can use `filter_height` and `filter_width`, if filters are not square.
    * `enable_maxpooling` specifies if the `?` in `[[CONV -> RELU]*N -> POOL?]*M` stands true. There should be `M` elements.
    * `pooling_stride` tells the network the strides it will use for each pooling layer. `M` elements in it.
      * There're also `pooling_stride_x` and `pooling_stride_y` for you
    * `pooling_size` is the sizes of the pooling window. There should be `M` elements.
      * Use `pooling_width` and `pooling_height` is pooling window is rectangular.
    * `padding_size` is the sizes for zero padding before each conv layer, so there should be `M*N` elements in it.
      * Again, `padding_width` and `padding_height` are there for you.
    * `enable_learning_rate_step_decay`, `enable_learning_rate_exponential_decay` and `enable_learning_rate_invert_t_decay` are parameters that enable learning rate decay. `learning_rate_decay_a0` and `learning_rate_decay_k` are parameters that tune the delay step. You can check [Annealing the learning rate from CS231n](http://cs231n.github.io/neural-networks-3/#anneal). A learning rate decay will be performed every `learning_rate_decay_unit` epochs.
    * `shuffle_training_samples` specified for every certain epochs, the input data set will be shuffled. `vertically_flip_training_samples` and `horizontally_flip_training_samples` tells the network if random horizontally flip and/or vertically flip input data will be performed during each shuffling.
  * Format for the 3D matrices file
    * Basically it's the same like 2D matrix file, only there might be more than one matrix, and each matrix will a depth dimension added.
    * [This file](test_data/mnist.txt) is the ASCII format of the first 10 samples from MNIST data set.
    * This is an example, of 2 3D matrices, and each matrix is 2 unit deep, 3 unit high, and 4 unit wide:
```
      X1 2 3 4
      x000 x001 x002 x003
      x010 x011 x012 x013
      x020 x021 x022 x023
      x100 x101 x102 x103
      x110 x111 x112 x113
      x120 x121 x122 x123
      X2 2 3 4
      x000 x001 x002 x003
      x010 x011 x012 x013
      x020 x021 x022 x023
      x100 x101 x102 x103
      x110 x111 x112 x113
      x120 x121 x122 x123
```
* Example
  * I created a simple CNN with MNIST data set, the network configuration file is [cnn.config](cnn.config).
  * As mentioned, the input data is [mnist.txt](test_data/mnist.txt), and the correct labels for these input data is [cnn_labels.txt](test_data/cnn_labels.txt). To make the data size smaller, the input data only contains first 10 images from MNIST.
  * Run the example with `./convolutional_network cnn.config` and you will see the results come out in a while.
* Note
  * The program is single-threaded, so it runs very slowly. I will try to add multi-threading to it, maybe.


