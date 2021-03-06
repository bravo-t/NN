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
    bool enable_learning_rate_step_decay;
    bool enable_learning_rate_exponential_decay;
    bool enable_learning_rate_invert_t_decay;
    int learning_rate_decay_unit;
    float learning_rate_decay_a0;
    float learning_rate_decay_k;
    int shuffle_training_samples;
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
    int save_checkpoint;
    char* params_save_dir;
    char* params_filename;
    char* mode;
    int number_of_threads;
} FCParameters;

typedef struct {
    ThreeDMatrix** X;
    int number_of_samples;
    float alpha;
    float learning_rate;
    int epochs;
    int minibatch_size;
    char* mode;
    /* Below parameters are used to config the convolutional network with pattern 
    INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC
    K is not used here, since the config of full connected network will be stored in fcnet_param
    */
    int N;
    int M; 
    // There will be N*M elements in below arrays related with filters
    int* filter_stride_x;
    int* filter_stride_y;
    int* filter_width;
    int* filter_height;
    int* filter_number;
    // There will be M elements in below arrays related with pooling layer
    bool* enable_maxpooling;
    int* pooling_stride_x;
    int* pooling_stride_y;
    int* pooling_width;
    int* pooling_height;
    int* padding_width;
    int* padding_height;
    bool verbose;
    int shuffle_training_samples;
    bool vertically_flip_training_samples;
    bool horizontally_flip_training_samples;
    bool enable_learning_rate_step_decay;
    bool enable_learning_rate_exponential_decay;
    bool enable_learning_rate_invert_t_decay;
    int learning_rate_decay_unit;
    float learning_rate_decay_a0;
    float learning_rate_decay_k;
    float learning_rate_decay_step;
    bool normalize_data_per_channel;
    bool use_rmsprop;
    float rmsprop_decay_rate;
    float rmsprop_eps;
    bool write_filters_as_images;
    int save_checkpoint;
    char* filter_image_dir;
    char* params_save_dir;
    char* params_filename;
    int number_of_threads;
    FCParameters* fcnet_param;
} ConvnetParameters;


#endif
