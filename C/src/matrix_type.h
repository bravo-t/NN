#ifndef __TYPE_HEADER__
#define __TYPE_HEADER__

typedef struct {
    int height;
    int width;
    float** d;
    bool initialized;
} TwoDMatrix;

typedef struct {
	int height;
    int width;
    int depth;
    float*** d;
    bool initialized;
} ThreeDMatrix;

typedef struct {
	ThreeDMatrix** X;
	ThreeDMatrix** correct_labels;
	int labels;
	float alpha;
	int epochs;
	
} ConvnetParameters;

typedef struct {
	TwoDMatrix* X; // Input data, height = minibatch_size, width = size_of_one_example
	// Below parameters are used in training
	TwoDMatrix* correct_labels; // Correct labels, height = minibatch_size, width = 1 
	int minibatch_size; // Size of the minibatch of training examples
	int labels; // How many scores will the nerual network calculate?
	float reg_strength; // Regulization strength
	float alpha; // the alpha in leaky ReLU, it should be set to 0 if ReLU is expected
	float learning_rate; // Step size of one value update
    int epochs;
	int network_depth; // Network depth
	int* hidden_layer_sizes; // Sizes of these hidden layers
    bool verbose;
    bool use_momentum_update;
    bool use_nag_update;
    bool use_rmsprop;
    float mu;
    float decay_rate;
    float eps;
    bool use_batchnorm;
    float batchnorm_momentum;
    float batchnorm_eps;
    char* params_save_dir;
    char* mode;
} FCParameters;

#endif
