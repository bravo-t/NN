#ifndef __FULLY_CONNECTED_HEADER__
#define __FULLY_CONNECTED_HEADER__


typedef struct {
    TwoDMatrix* X; // Input data, height = minibatch_size, width = size_of_one_example
    // Below parameters are used in training
    TwoDMatrix* correct_labels; // Correct labels, height = minibatch_size, width = 1 
    int minibatch_size; // Size of the minibatch of training examples
    int labels; // How many scores will the nerual network calculate?
    float reg_strength; // Regulization strength
    float alpha; // the alpha in leaky ReLU, it should be set to 0 if ReLU is expected
    float learning_rate; // Step size of one value update
    int network_depth; // Network depth
    int* hidden_layer_sizes; // Sizes of these hidden layers
} parameters;


#endif