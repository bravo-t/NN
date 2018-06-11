#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <malloc.h>
#include <math.h>
#include <string.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>
#include "thread_barrier.h"
#include "thread_control.h"
#include "network_type.h"
#include "matrix_operations.h"
#include "matrix_operations_multithread.h"
#include "layers.h"
#include "misc_utils.h"
#include "fully_connected_net.h"
#include "fully_connected_net_multithread.h"
#include "convnet_operations.h"
#include "convnet_layers.h"
#include "convnet.h"
#include "convnet_multithread.h"

#if defined(DEBUG) && DEBUG > 1
    pthread_mutex_t debug_mutex = PTHREAD_MUTEX_INITIALIZER;
#endif


int trainConvnet_multithread(ConvnetParameters* network_params) {
    ThreeDMatrix** training_data = network_params->X;
    int number_of_samples = network_params->number_of_samples;
    int minibatch_size = network_params->minibatch_size;
    /* INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC */
    int N = network_params->N;
    int M = network_params->M;
    int* filter_stride_x = network_params->filter_stride_x;
    int* filter_stride_y = network_params->filter_stride_y;
    int* filter_width = network_params->filter_width;
    int* filter_height = network_params->filter_height;
    int* filter_number = network_params->filter_number;
    bool* enable_maxpooling = network_params->enable_maxpooling;
    int* pooling_stride_x = network_params->pooling_stride_x;
    int* pooling_stride_y = network_params->pooling_stride_y;
    int* pooling_width = network_params->pooling_width;
    int* pooling_height = network_params->pooling_height;
    int epochs = network_params->epochs;
    int* padding_width = network_params->padding_width;
    int* padding_height = network_params->padding_height;
    float alpha = network_params->alpha;
    float learning_rate = network_params->learning_rate;
    float base_learning_rate = learning_rate;
    bool verbose = network_params->verbose;
    
    int shuffle_training_samples = network_params->shuffle_training_samples;
    bool vertically_flip_training_samples = network_params->vertically_flip_training_samples;
    bool horizontally_flip_training_samples = network_params->horizontally_flip_training_samples;

    bool use_rmsprop = network_params->use_rmsprop;
    float rmsprop_decay_rate = network_params->rmsprop_decay_rate;
    float rmsprop_eps = network_params->rmsprop_eps;

    bool normalize_data_per_channel = network_params->normalize_data_per_channel;

    bool write_filters_as_images = network_params->write_filters_as_images;
    char* filter_image_dir = network_params->filter_image_dir;
    
    bool enable_learning_rate_step_decay = network_params->enable_learning_rate_step_decay;
    bool enable_learning_rate_exponential_decay = network_params->enable_learning_rate_exponential_decay;
    bool enable_learning_rate_invert_t_decay = network_params->enable_learning_rate_invert_t_decay;
    int learning_rate_decay_unit = network_params->learning_rate_decay_unit;
    float learning_rate_decay_a0 = network_params->learning_rate_decay_a0;
    float learning_rate_decay_k = network_params->learning_rate_decay_k;
    char* param_dir = network_params->params_save_dir;
    int save_checkpoint = network_params->save_checkpoint;

    int number_of_threads = network_params->number_of_threads;

    // Turn these features off to reduce the complexity for now
    network_params->fcnet_param->use_momentum_update = false;
    network_params->fcnet_param->use_batchnorm = false;
    network_params->fcnet_param->use_nag_update = false;
    //float current_fcnet_learning_rate = network_params->fcnet_param->learning_rate;
    network_params->fcnet_param->enable_learning_rate_step_decay = enable_learning_rate_step_decay;
    network_params->fcnet_param->enable_learning_rate_exponential_decay = enable_learning_rate_exponential_decay;
    network_params->fcnet_param->enable_learning_rate_invert_t_decay = enable_learning_rate_invert_t_decay;
    network_params->fcnet_param->learning_rate_decay_unit = learning_rate_decay_unit;
    network_params->fcnet_param->learning_rate_decay_a0 = learning_rate_decay_a0;
    network_params->fcnet_param->learning_rate_decay_k = learning_rate_decay_k;
    
    printf("CONVNET INFO: Convolutional neural network started with %d CPU cores\n",number_of_threads);
    if (normalize_data_per_channel) {
        printf("CONVNET INFO: Normalizing input data\n");
        for(int i=0;i<number_of_samples;i++) {
            normalize3DMatrixPerDepth(training_data[i], training_data[i]);
        }
    }
    
    if (use_rmsprop) {
        printf("CONVNET INFO: RMSProp is enabled.\n");
    }

    printf("CONVNET INFO: Initializing learnable weights and intermediate layers\n");
    unsigned long long int total_parameters = 0;
    unsigned long long int total_memory = 0;
    float** losses = malloc(sizeof(float)*number_of_threads);
    /*
    C will hold intermediate values of CONV -> RELU layer, C[M][N][minibatch_size]
    P will hold intermediate values of POOL, P[M][minibatch_size]
    F will be a 2D array that contains filters, F[M][N][filter_number]
    b will be a 2D array that holds biases, b[M][N][filter_number]
    */
    ThreeDMatrix**** C = malloc(sizeof(ThreeDMatrix***)*M);
    ThreeDMatrix**** dC = malloc(sizeof(ThreeDMatrix***)*M);
    ThreeDMatrix*** P = malloc(sizeof(ThreeDMatrix**)*M);
    ThreeDMatrix*** dP = malloc(sizeof(ThreeDMatrix**)*M);
    ThreeDMatrix**** F = malloc(sizeof(ThreeDMatrix***)*M);
    ThreeDMatrix**** dF = malloc(sizeof(ThreeDMatrix***)*M);
    ThreeDMatrix**** b = malloc(sizeof(ThreeDMatrix***)*M);
    ThreeDMatrix**** db = malloc(sizeof(ThreeDMatrix***)*M);
    ThreeDMatrix**** Fcache = NULL;
    ThreeDMatrix**** bcache = NULL;
    if (use_rmsprop) {
        Fcache = (ThreeDMatrix****) malloc(sizeof(ThreeDMatrix***)*M);
        bcache = (ThreeDMatrix****) malloc(sizeof(ThreeDMatrix***)*M);
    }
    TwoDMatrix* dP2D = matrixMalloc(sizeof(TwoDMatrix));

    ThreeDMatrix** dX = malloc(sizeof(ThreeDMatrix*)*minibatch_size);
    for(int i=0;i<minibatch_size;i++) {
        dX[i] = matrixMalloc(sizeof(ThreeDMatrix));
        init3DMatrix(dX[i], training_data[i]->depth, training_data[i]->height, training_data[i]->width);
    }

    int layer_data_depth = training_data[0]->depth;
    int layer_data_height = training_data[0]->height;
    int layer_data_width = training_data[0]->width;
    printf("CONVNET INFO: INPUT: [%dx%dx%d]\t\t\tweights: 0\n",layer_data_width,layer_data_height,layer_data_depth);
    total_memory += layer_data_depth*layer_data_height*layer_data_width;
    for(int i=0;i<M;i++) {
        C[i] = (ThreeDMatrix***) malloc(sizeof(ThreeDMatrix**)*N);
        F[i] = (ThreeDMatrix***) malloc(sizeof(ThreeDMatrix**)*N);
        b[i] = (ThreeDMatrix***) malloc(sizeof(ThreeDMatrix**)*N);
        dC[i] = (ThreeDMatrix***) malloc(sizeof(ThreeDMatrix**)*N);
        dF[i] = (ThreeDMatrix***) malloc(sizeof(ThreeDMatrix**)*N);
        db[i] = (ThreeDMatrix***) malloc(sizeof(ThreeDMatrix**)*N);
        if (use_rmsprop) {
            Fcache[i] = (ThreeDMatrix***) malloc(sizeof(ThreeDMatrix**)*N);
            bcache[i] = (ThreeDMatrix***) malloc(sizeof(ThreeDMatrix**)*N);
        }
        for(int j=0;j<N;j++) {
            F[i][j] = (ThreeDMatrix**) malloc(sizeof(ThreeDMatrix*)*filter_number[i*N+j]);
            b[i][j] = (ThreeDMatrix**) malloc(sizeof(ThreeDMatrix*)*filter_number[i*N+j]);
            C[i][j] = (ThreeDMatrix**) malloc(sizeof(ThreeDMatrix*)*minibatch_size);
            dF[i][j] = (ThreeDMatrix**) malloc(sizeof(ThreeDMatrix*)*filter_number[i*N+j]);
            db[i][j] = (ThreeDMatrix**) malloc(sizeof(ThreeDMatrix*)*filter_number[i*N+j]);
            dC[i][j] = (ThreeDMatrix**) malloc(sizeof(ThreeDMatrix*)*minibatch_size);
            if (use_rmsprop) {
                Fcache[i][j] = (ThreeDMatrix**) malloc(sizeof(ThreeDMatrix*)*filter_number[i*N+j]);
                bcache[i][j] = (ThreeDMatrix**) malloc(sizeof(ThreeDMatrix*)*filter_number[i*N+j]);
            }
            for(int k=0;k<filter_number[i*N+j];k++) {
                F[i][j][k] = matrixMalloc(sizeof(ThreeDMatrix));
                b[i][j][k] = matrixMalloc(sizeof(ThreeDMatrix));
                dF[i][j][k] = matrixMalloc(sizeof(ThreeDMatrix));
                db[i][j][k] = matrixMalloc(sizeof(ThreeDMatrix));
                //init3DMatrixNormRand(F[i][j][k],layer_data_depth,filter_height[i*N+j],filter_width[i*N+j],0.0,1.0,sqrt(layer_data_height*layer_data_width*layer_data_depth));
                init3DMatrixNormRand(F[i][j][k],layer_data_depth,filter_height[i*N+j],filter_width[i*N+j],0.0,1.0,layer_data_height*layer_data_width*layer_data_depth);
                init3DMatrix(b[i][j][k],1,1,1);
                init3DMatrix(dF[i][j][k],layer_data_depth,filter_height[i*N+j],filter_width[i*N+j]);
                init3DMatrix(db[i][j][k],1,1,1);
                if (use_rmsprop) {
                    Fcache[i][j][k] = matrixMalloc(sizeof(ThreeDMatrix));
                    bcache[i][j][k] = matrixMalloc(sizeof(ThreeDMatrix));
                    init3DMatrix(Fcache[i][j][k],layer_data_depth,filter_height[i*N+j],filter_width[i*N+j]);
                    init3DMatrix(bcache[i][j][k],1,1,1);
                }
            }
            int filter_depth = layer_data_depth;
            layer_data_depth = filter_number[i*N+j];
            layer_data_height = calcOutputSize(layer_data_height,padding_height[i*N+j],filter_height[i*N+j],filter_stride_y[i*N+j]);
            layer_data_width = calcOutputSize(layer_data_width,padding_width[i*N+j],filter_width[i*N+j],filter_stride_x[i*N+j]);
            for(int l=0;l<minibatch_size;l++) {
                C[i][j][l] = matrixMalloc(sizeof(ThreeDMatrix));
                dC[i][j][l] = matrixMalloc(sizeof(ThreeDMatrix));
                init3DMatrix(C[i][j][l], layer_data_depth, layer_data_height, layer_data_width);
                init3DMatrix(dC[i][j][l], layer_data_depth, layer_data_height, layer_data_width);
            }
            printf("CONVNET INFO: CONV[%dx%d-%d]: [%dx%dx%d]\t\tweights: (%d*%d*%d)*%d=%d\n",
                filter_width[i*N+j],filter_height[i*N+j],filter_depth, layer_data_width,layer_data_height,layer_data_depth,
                filter_width[i*N+j],filter_height[i*N+j],filter_depth,layer_data_depth,filter_width[i*N+j]*filter_height[i*N+j]*filter_depth*layer_data_depth);
            total_memory += layer_data_depth*layer_data_height*layer_data_width;
            total_parameters += filter_width[i*N+j]*filter_height[i*N+j]*filter_depth*layer_data_depth;
        }
        if (enable_maxpooling[i]) {
            P[i] = (ThreeDMatrix**) malloc(sizeof(ThreeDMatrix*)*minibatch_size);
            dP[i] = (ThreeDMatrix**) malloc(sizeof(ThreeDMatrix*)*minibatch_size);
            layer_data_height = calcOutputSize(layer_data_height,0,pooling_height[i],pooling_stride_y[i]);
            layer_data_width = calcOutputSize(layer_data_width,0,pooling_width[i],pooling_stride_x[i]);
            for(int m=0;m<minibatch_size;m++) {
                P[i][m] = matrixMalloc(sizeof(ThreeDMatrix));
                dP[i][m] = matrixMalloc(sizeof(ThreeDMatrix));
                init3DMatrix(P[i][m],layer_data_depth,layer_data_height,layer_data_width);
                init3DMatrix(dP[i][m],layer_data_depth,layer_data_height,layer_data_width);
            }
            printf("CONVNET INFO: POOL[%dx%d]: [%dx%dx%d]\t\tweights: 0\n",pooling_width[i],pooling_height[i],layer_data_width,layer_data_height,layer_data_depth);
            total_memory += layer_data_depth*layer_data_height*layer_data_width;
        } else {
            P[i] = C[i][N-1];
            dP[i] = dC[i][N-1];
        }
    }
    //for(int i=0;i<number_of_samples;i++) {
    //    dP3D[i] = matrixMalloc(sizeof(ThreeDMatrix));
    //    init3DMatrix(dP3D[i],P[M-1][i]->depth, P[M-1][i]->height, P[M-1][i]->width);
    //}
    // Initialize the fully connected network in convnet
    int* fcnet_hidden_layer_sizes = network_params->fcnet_param->hidden_layer_sizes;
    int K = network_params->fcnet_param->network_depth;
    TwoDMatrix** Ws = malloc(sizeof(TwoDMatrix*)*K);
    TwoDMatrix** bs = malloc(sizeof(TwoDMatrix*)*K);
    TwoDMatrix** Wscache = NULL;
    TwoDMatrix** bscache = NULL;
    if (use_rmsprop) {
        Wscache = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*K);
        bscache = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*K);
    }
    int fcnet_labels = network_params->fcnet_param->labels;
    fcnet_hidden_layer_sizes[K-1] = fcnet_labels;
    printf("FCNET INFO: INPUT[%dx%d]\t\t\tweights: 0\n",minibatch_size,layer_data_depth*layer_data_height*layer_data_width);
    total_memory += layer_data_depth*layer_data_height*layer_data_width;
    int former_width = layer_data_depth*layer_data_height*layer_data_width;
    for(int i=0;i<K;i++) {
        // Initialize layer data holders
        Ws[i] = matrixMalloc(sizeof(TwoDMatrix));
        bs[i] = matrixMalloc(sizeof(TwoDMatrix));
        init2DMatrixNormRand(Ws[i],former_width,fcnet_hidden_layer_sizes[i],0.0,1.0, former_width);
        init2DMatrixZero(bs[i],1,fcnet_hidden_layer_sizes[i]);
        if (use_rmsprop) {
            Wscache[i] = matrixMalloc(sizeof(TwoDMatrix));
            bscache[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrixZero(Wscache[i],former_width,fcnet_hidden_layer_sizes[i]);
            init2DMatrixZero(bscache[i],1,fcnet_hidden_layer_sizes[i]);
        }
        printf("FCNET INFO: FC[%dx%dx%d]\t\t\tweights: %d*%d=%d\n",1,1,fcnet_hidden_layer_sizes[i],former_width,fcnet_hidden_layer_sizes[i],former_width*fcnet_hidden_layer_sizes[i]);
        total_parameters += former_width*fcnet_hidden_layer_sizes[i];
        // Initialize variables for optimization
        /*
        if (false) {
            gammas[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrix(gammas[i],1,fcnet_hidden_layer_sizes[i]);
            betas[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrix(betas[i],1,fcnet_hidden_layer_sizes[i]);
            means[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrix(means[i],1,fcnet_hidden_layer_sizes[i]);
            vars[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrix(vars[i],1,fcnet_hidden_layer_sizes[i]);
        }
        */
        former_width = fcnet_hidden_layer_sizes[i];
    }
    // Babysit the fully connected network core
    network_params->fcnet_param->minibatch_size = minibatch_size;
    network_params->fcnet_param->epochs = 1;
    // Print some statistical info
    printf("CONVNET INFO: Total parameters: %lld\n",total_parameters);
    char memory_unit_per_image = determineMemoryUnit(total_memory*sizeof(float));
    float memory_usage_per_image = memoryUsageReadable(total_memory*sizeof(float),memory_unit_per_image);
    char memory_unit_total = determineMemoryUnit(total_memory*sizeof(float)*minibatch_size);
    float memory_usage_total = memoryUsageReadable(total_memory*sizeof(float)*minibatch_size,memory_unit_total);
    printf("CONVNET INFO: Memory usage: %f%cB per image, total memory: %f%cB\n",memory_usage_per_image, memory_unit_per_image, memory_usage_total, memory_unit_total);
    
    ThreeDMatrix** CONV_OUT = NULL;

    // Start training the network
    /*
    C will hold intermediate values of CONV -> RELU layer, C[M][N][minibatch_size]
    P will hold intermediate values of POOL, P[M][minibatch_size]
    F will be a 2D array that contains filters, F[M][N][filter_number]
    b will be a 2D array that holds biases, b[M][N][filter_number]
    */
    /* INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC */
    //ThreeDMatrix* dX = matrixMalloc(sizeof(ThreeDMatrix));
    //init3DMatrix(dX, training_data->depth, training_data->height, training_data->width);
    printf("CONVNET INFO: Creating slave threads...\n");
    // slave threads for convnet
    pthread_mutex_t forward_prop_control_handle_mutex = PTHREAD_MUTEX_INITIALIZER;
    thread_barrier_t forward_prop_inst_ready = THREAD_BARRIER_INITIALIZER;
    thread_barrier_t forward_prop_inst_ack = THREAD_BARRIER_INITIALIZER;
    thread_barrier_t forward_prop_thread_complete = THREAD_BARRIER_INITIALIZER;
    ThreadControl* forward_prop_control_handle = initControlHandle(&forward_prop_control_handle_mutex, &forward_prop_inst_ready, &forward_prop_inst_ack, &forward_prop_thread_complete, number_of_threads);

    pthread_mutex_t backward_prop_control_handle_mutex = PTHREAD_MUTEX_INITIALIZER;
    thread_barrier_t backward_prop_inst_ready = THREAD_BARRIER_INITIALIZER;
    thread_barrier_t backward_prop_inst_ack = THREAD_BARRIER_INITIALIZER;
    thread_barrier_t backward_prop_thread_complete = THREAD_BARRIER_INITIALIZER;
    ThreadControl* backward_prop_control_handle = initControlHandle(&backward_prop_control_handle_mutex, &backward_prop_inst_ready, &backward_prop_inst_ack, &backward_prop_thread_complete, number_of_threads);
    
    pthread_mutex_t update_weights_control_handle_mutex = PTHREAD_MUTEX_INITIALIZER;
    thread_barrier_t update_weights_inst_ready = THREAD_BARRIER_INITIALIZER;
    thread_barrier_t update_weights_inst_ack = THREAD_BARRIER_INITIALIZER;
    thread_barrier_t update_weights_thread_complete = THREAD_BARRIER_INITIALIZER;
    ThreadControl* update_weights_control_handle = initControlHandle(&update_weights_control_handle_mutex, &update_weights_inst_ready, &update_weights_inst_ack, &update_weights_thread_complete, number_of_threads);

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    pthread_t* forward_prop = malloc(sizeof(pthread_t)*number_of_threads);
    pthread_t* backward_prop = malloc(sizeof(pthread_t)*number_of_threads);
    pthread_t* update_weights = malloc(sizeof(pthread_t)*number_of_threads);

    ConvnetSlaveArgs** forward_prop_arguments = malloc(sizeof(ConvnetSlaveArgs*)*number_of_threads);
    ConvnetSlaveArgs** backward_prop_arguments = malloc(sizeof(ConvnetSlaveArgs*)*number_of_threads);
    ConvnetSlaveArgs** update_weights_arguments = malloc(sizeof(ConvnetSlaveArgs*)*number_of_threads);

    for(int i=0;i<number_of_threads;i++) {
        forward_prop_arguments[i] = (ConvnetSlaveArgs*) malloc(sizeof(ConvnetSlaveArgs));
        backward_prop_arguments[i] = (ConvnetSlaveArgs*) malloc(sizeof(ConvnetSlaveArgs));
        update_weights_arguments[i] = (ConvnetSlaveArgs*) malloc(sizeof(ConvnetSlaveArgs));

        assignConvSlaveArguments(forward_prop_arguments[i], 
            M, N, minibatch_size, NULL, NULL, &CONV_OUT,
            C, P, F, b, dC, dP, dF, db, Fcache, bcache,
            filter_number, filter_height, filter_width, filter_stride_y, filter_stride_x, 
            padding_width, padding_height, 
            enable_maxpooling, pooling_height, pooling_width, pooling_stride_x, pooling_stride_y, 
            learning_rate, use_rmsprop, rmsprop_decay_rate, rmsprop_eps,
            alpha, verbose, i, number_of_threads, forward_prop_control_handle);
        assignConvSlaveArguments(backward_prop_arguments[i], 
            M, N, minibatch_size, training_data, dX, &CONV_OUT,
            C, P, F, b, dC, dP, dF, db, Fcache, bcache,
            filter_number, filter_height, filter_width, filter_stride_y, filter_stride_x, 
            padding_width, padding_height, 
            enable_maxpooling, pooling_height, pooling_width, pooling_stride_x, pooling_stride_y, 
            learning_rate, use_rmsprop, rmsprop_decay_rate, rmsprop_eps,
            alpha, verbose, i, number_of_threads, backward_prop_control_handle);
        assignConvSlaveArguments(update_weights_arguments[i], 
            M, N, minibatch_size, NULL, NULL, &CONV_OUT,
            C, P, F, b, dC, dP, dF, db, Fcache, bcache,
            filter_number, filter_height, filter_width, filter_stride_y, filter_stride_x, 
            padding_width, padding_height, 
            enable_maxpooling, pooling_height, pooling_width, pooling_stride_x, pooling_stride_y, 
            learning_rate, use_rmsprop, rmsprop_decay_rate, rmsprop_eps,
            alpha, verbose, i, number_of_threads, update_weights_control_handle);

        int create_thread_error;
        /*
        forward_prop_arguments[i] is the type of SlaveArgs*, which is expected by FCNET_forwardPropagation_slave.
        However while creating slave threads, I used &forward_prop_arguments[i], this is a type of SlaveArgs**.
        So the "&" is not needed.
        */
        create_thread_error = pthread_create(&forward_prop[i],&attr,CONV_forwardPropagation_slave,forward_prop_arguments[i]);
        if (create_thread_error) {
            printf("Error happened while creating slave threads\n");
            exit(-1);
        }

        create_thread_error = pthread_create(&backward_prop[i],&attr,CONV_backwardPropagation_slave,backward_prop_arguments[i]);
        if (create_thread_error) {
            printf("Error happened while creating slave threads\n");
            exit(-1);
        }
        create_thread_error = pthread_create(&update_weights[i],&attr,CONV_updateWeights_slave,update_weights_arguments[i]);
        if (create_thread_error) {
            printf("Error happened while creating slave threads\n");
            exit(-1);
        }
    }

    // Now slave threads for fully connected network
    float fcnet_reg_strength = network_params->fcnet_param->reg_strength;
    int fcnet_network_depth = network_params->fcnet_param->network_depth;
    
    //float fcnet_mu = network_params->fcnet_param->mu; // or 0.5,0.95, 0.99
    float fcnet_decay_rate = network_params->fcnet_param->decay_rate; // or with more 9s in it
    float fcnet_eps = network_params->fcnet_param->eps;
    //TwoDMatrix* fcnet_training_data = network_params->fcnet_param->X;
    // Initialize all learnable parameters
    // Hidden layers
    TwoDMatrix** Hs = malloc(sizeof(TwoDMatrix*)*fcnet_network_depth);
    // Gradient descend values of Weights
    TwoDMatrix** dWs = malloc(sizeof(TwoDMatrix*)*fcnet_network_depth);
    // Gradient descend values of Biases
    TwoDMatrix** dbs = malloc(sizeof(TwoDMatrix*)*fcnet_network_depth);
    // Gradient descend values of Hidden layers
    TwoDMatrix** dHs = malloc(sizeof(TwoDMatrix*)*fcnet_network_depth);
    // Below variables are used in optimization algorithms
    // Batch normalization layers;
    TwoDMatrix** dgammas = NULL;
    TwoDMatrix** dbetas = NULL;
    TwoDMatrix** means = NULL;
    TwoDMatrix** vars = NULL;
    TwoDMatrix** Hs_normalized = NULL;

    if (network_params->fcnet_param->use_batchnorm) {
        dgammas = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*fcnet_network_depth);
        dbetas = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*fcnet_network_depth);
        means = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*fcnet_network_depth);
        vars = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*fcnet_network_depth);
        Hs_normalized = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*fcnet_network_depth);
    }

    for(int i=0;i<fcnet_network_depth;i++) {
        // Initialize layer data holders
        Hs[i] = matrixMalloc(sizeof(TwoDMatrix));
        dWs[i] = matrixMalloc(sizeof(TwoDMatrix));
        dbs[i] = matrixMalloc(sizeof(TwoDMatrix));
        dHs[i] = matrixMalloc(sizeof(TwoDMatrix));
        init2DMatrix(Hs[i],minibatch_size,fcnet_hidden_layer_sizes[i]);

        // Initialize variables for optimization
        if (network_params->fcnet_param->use_batchnorm) {
            dgammas[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrix(dgammas[i],1,fcnet_hidden_layer_sizes[i]);
            dbetas[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrix(dbetas[i],1,fcnet_hidden_layer_sizes[i]);
            means[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrix(means[i],1,fcnet_hidden_layer_sizes[i]);
            vars[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrix(vars[i],1,fcnet_hidden_layer_sizes[i]);
            Hs_normalized[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrixZero(Hs_normalized[i],minibatch_size,fcnet_hidden_layer_sizes[i]);
        }
        //former_width = hidden_layer_sizes[i];
    }
    
    // Create slave workers
    bool fcnet_forward_prop_mem_alloc = false;
    thread_barrier_t fcnet_forward_prop_barrier = THREAD_BARRIER_INITIALIZER;
    thread_barrier_init(&fcnet_forward_prop_barrier,number_of_threads);
    pthread_mutex_t fcnet_forward_prop_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t fcnet_forward_prop_cond = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t fcnet_forward_prop_control_handle_mutex = PTHREAD_MUTEX_INITIALIZER;
    thread_barrier_t fcnet_forward_prop_inst_ready = THREAD_BARRIER_INITIALIZER;
    thread_barrier_t fcnet_forward_prop_inst_ack = THREAD_BARRIER_INITIALIZER;
    thread_barrier_t fcnet_forward_prop_thread_complete = THREAD_BARRIER_INITIALIZER;
    //thread_barrier_init(&forward_prop_inst_ready,number_of_threads);
    //thread_barrier_init(&forward_prop_inst_ack,number_of_threads);
    ThreadControl* fcnet_forward_prop_control_handle = initControlHandle(&fcnet_forward_prop_control_handle_mutex, 
        &fcnet_forward_prop_inst_ready, &fcnet_forward_prop_inst_ack, &fcnet_forward_prop_thread_complete, number_of_threads);

    bool fcnet_calc_loss_mem_alloc = false;
    thread_barrier_t fcnet_calc_loss_barrier = THREAD_BARRIER_INITIALIZER;
    thread_barrier_init(&fcnet_calc_loss_barrier,number_of_threads);
    pthread_mutex_t fcnet_calc_loss_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t fcnet_calc_loss_cond = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t fcnet_calc_loss_control_handle_mutex = PTHREAD_MUTEX_INITIALIZER;
    thread_barrier_t fcnet_calc_loss_inst_ready = THREAD_BARRIER_INITIALIZER;
    thread_barrier_t fcnet_calc_loss_inst_ack = THREAD_BARRIER_INITIALIZER;
    thread_barrier_t fcnet_calc_loss_thread_complete = THREAD_BARRIER_INITIALIZER;
    //thread_barrier_init(&calc_loss_inst_ready,number_of_threads);
    //thread_barrier_init(&calc_loss_inst_ack,number_of_threads);
    ThreadControl* fcnet_calc_loss_control_handle = initControlHandle(&fcnet_calc_loss_control_handle_mutex, 
        &fcnet_calc_loss_inst_ready, &fcnet_calc_loss_inst_ack, &fcnet_calc_loss_thread_complete, number_of_threads);
    
    bool fcnet_backward_prop_mem_alloc = false;
    thread_barrier_t fcnet_backward_prop_barrier = THREAD_BARRIER_INITIALIZER;
    thread_barrier_init(&fcnet_backward_prop_barrier,number_of_threads);
    pthread_mutex_t fcnet_backward_prop_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t fcnet_backward_prop_cond = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t fcnet_backward_prop_control_handle_mutex = PTHREAD_MUTEX_INITIALIZER;
    thread_barrier_t fcnet_backward_prop_inst_ready = THREAD_BARRIER_INITIALIZER;
    thread_barrier_t fcnet_backward_prop_inst_ack = THREAD_BARRIER_INITIALIZER;
    thread_barrier_t fcnet_backward_prop_thread_complete = THREAD_BARRIER_INITIALIZER;
    //thread_barrier_init(&backward_prop_inst_ready,number_of_threads);
    //thread_barrier_init(&backward_prop_inst_ack,number_of_threads);
    ThreadControl* fcnet_backward_prop_control_handle = initControlHandle(&fcnet_backward_prop_control_handle_mutex, 
        &fcnet_backward_prop_inst_ready, &fcnet_backward_prop_inst_ack, &fcnet_backward_prop_thread_complete, number_of_threads);
    
    bool fcnet_update_weights_mem_alloc = false;
    thread_barrier_t fcnet_update_weights_barrier = THREAD_BARRIER_INITIALIZER;
    thread_barrier_init(&fcnet_update_weights_barrier,number_of_threads);
    pthread_mutex_t fcnet_update_weights_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t fcnet_update_weights_cond = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t fcnet_update_weights_control_handle_mutex = PTHREAD_MUTEX_INITIALIZER;
    thread_barrier_t fcnet_update_weights_inst_ready = THREAD_BARRIER_INITIALIZER;
    thread_barrier_t fcnet_update_weights_inst_ack = THREAD_BARRIER_INITIALIZER;
    thread_barrier_t fcnet_update_weights_thread_complete = THREAD_BARRIER_INITIALIZER;
    //thread_barrier_init(&update_weights_inst_ready,number_of_threads);
    //thread_barrier_init(&update_weights_inst_ack,number_of_threads);
    ThreadControl* fcnet_update_weights_control_handle = initControlHandle(&fcnet_update_weights_control_handle_mutex, 
        &fcnet_update_weights_inst_ready, &fcnet_update_weights_inst_ack, &fcnet_update_weights_thread_complete, number_of_threads);
    
    pthread_t* fcnet_forward_prop = malloc(sizeof(pthread_t)*number_of_threads);
    pthread_t* fcnet_calc_loss = malloc(sizeof(pthread_t)*number_of_threads);
    pthread_t* fcnet_backward_prop = malloc(sizeof(pthread_t)*number_of_threads);
    pthread_t* fcnet_update_weights = malloc(sizeof(pthread_t)*number_of_threads);

    SlaveArgs** fcnet_forward_prop_arguments = malloc(sizeof(SlaveArgs*)*number_of_threads);
    SlaveArgs** fcnet_calc_loss_arguments = malloc(sizeof(SlaveArgs*)*number_of_threads);
    SlaveArgs** fcnet_backward_prop_arguments = malloc(sizeof(SlaveArgs*)*number_of_threads);
    SlaveArgs** fcnet_update_weights_arguments = malloc(sizeof(SlaveArgs*)*number_of_threads);

    TwoDMatrix* X = matrixMalloc(sizeof(TwoDMatrix));

    for(int i=0;i<number_of_threads;i++) {
        losses[i] = malloc(sizeof(float)*2);
        
        fcnet_forward_prop_arguments[i] = (SlaveArgs*) malloc(sizeof(SlaveArgs));
        fcnet_calc_loss_arguments[i] = (SlaveArgs*) malloc(sizeof(SlaveArgs));
        fcnet_backward_prop_arguments[i] = (SlaveArgs*) malloc(sizeof(SlaveArgs));
        fcnet_update_weights_arguments[i] = (SlaveArgs*) malloc(sizeof(SlaveArgs));

        assignSlaveArguments(fcnet_forward_prop_arguments[i], 
            fcnet_forward_prop_control_handle,
            i,
            fcnet_network_depth,
            X,
            Ws,
            bs,
            Hs,
            dWs,
            dbs,
            dHs,
            Wscache,
            bscache,
            network_params->fcnet_param->correct_labels,
            NULL,
            alpha,
            &learning_rate,
            fcnet_reg_strength,
            fcnet_decay_rate,
            fcnet_eps,
            use_rmsprop,
            &fcnet_forward_prop_mem_alloc,
            number_of_threads,
            &fcnet_forward_prop_mutex,
            &fcnet_forward_prop_cond,
            &fcnet_forward_prop_barrier,
            NULL,
            NULL);
        assignSlaveArguments(fcnet_calc_loss_arguments[i], 
            fcnet_calc_loss_control_handle,
            i,
            fcnet_network_depth,
            X,
            Ws,
            bs,
            Hs,
            dWs,
            dbs,
            dHs,
            Wscache,
            bscache,
            network_params->fcnet_param->correct_labels,
            NULL,
            alpha,
            &learning_rate,
            fcnet_reg_strength,
            fcnet_decay_rate,
            fcnet_eps,
            use_rmsprop,
            &fcnet_calc_loss_mem_alloc,
            number_of_threads,
            &fcnet_calc_loss_mutex,
            &fcnet_calc_loss_cond,
            &fcnet_calc_loss_barrier,
            NULL,
            losses[i]);
        assignSlaveArguments(fcnet_backward_prop_arguments[i], 
            fcnet_backward_prop_control_handle,
            i,
            fcnet_network_depth,
            X,
            Ws,
            bs,
            Hs,
            dWs,
            dbs,
            dHs,
            Wscache,
            bscache,
            network_params->fcnet_param->correct_labels,
            dP2D,
            alpha,
            &learning_rate,
            fcnet_reg_strength,
            fcnet_decay_rate,
            fcnet_eps,
            use_rmsprop,
            &fcnet_backward_prop_mem_alloc,
            number_of_threads,
            &fcnet_backward_prop_mutex,
            &fcnet_backward_prop_cond,
            &fcnet_backward_prop_barrier,
            NULL,
            NULL);
        assignSlaveArguments(fcnet_update_weights_arguments[i], 
            fcnet_update_weights_control_handle,
            i,
            fcnet_network_depth,
            X,
            Ws,
            bs,
            Hs,
            dWs,
            dbs,
            dHs,
            Wscache,
            bscache,
            network_params->fcnet_param->correct_labels,
            NULL,
            alpha,
            &learning_rate,
            fcnet_reg_strength,
            fcnet_decay_rate,
            fcnet_eps,
            use_rmsprop,
            &fcnet_update_weights_mem_alloc,
            number_of_threads,
            &fcnet_update_weights_mutex,
            &fcnet_update_weights_cond,
            &fcnet_update_weights_barrier,
            NULL,
            NULL);

        int create_thread_error;
        /*
        fcnet_forward_prop_arguments[i] is the type of SlaveArgs*, which is expected by FCNET_forwardPropagation_slave.
        However while creating slave threads, I used &fcnet_forward_prop_arguments[i], this is a type of SlaveArgs**.
        So the "&" is not needed.
        */
        create_thread_error = pthread_create(&fcnet_forward_prop[i],&attr,FCNET_forwardPropagation_slave,fcnet_forward_prop_arguments[i]);
        if (create_thread_error) {
            printf("Error happened while creating slave threads\n");
            exit(-1);
        }
        create_thread_error = pthread_create(&fcnet_calc_loss[i],&attr,FCNET_calcLoss_slave,fcnet_calc_loss_arguments[i]);
        if (create_thread_error) {
            printf("Error happened while creating slave threads\n");
            exit(-1);
        }
        create_thread_error = pthread_create(&fcnet_backward_prop[i],&attr,FCNET_backwardPropagation_slave,fcnet_backward_prop_arguments[i]);
        if (create_thread_error) {
            printf("Error happened while creating slave threads\n");
            exit(-1);
        }
        create_thread_error = pthread_create(&fcnet_update_weights[i],&attr,FCNET_updateWeights_slave,fcnet_update_weights_arguments[i]);
        if (create_thread_error) {
            printf("Error happened while creating slave threads\n");
            exit(-1);
        }
    }




    printf("CONVNET INFO: Training network...\n");
    for(int e=1;e<=epochs;e++) {
        learning_rate = decayLearningRate(enable_learning_rate_step_decay,
            enable_learning_rate_exponential_decay,
            enable_learning_rate_invert_t_decay,
            learning_rate_decay_unit,
            learning_rate_decay_k,
            learning_rate_decay_a0,
            e,
            base_learning_rate,
            learning_rate);

        int iterations = number_of_samples/minibatch_size;
        float total_data_loss = 0.0f;
        float total_reg_loss = 0.0f;
        float training_accu = 0.0f;
        for(int iter=0;iter <iterations; iter++) {
            CONV_OUT = training_data + iter*sizeof(ThreeDMatrix*);
            // Forward propagation
            threadController_master(forward_prop_control_handle, THREAD_RESUME);
    
            // Feed data to fully connected network
            init2DMatrix(X,minibatch_size,layer_data_depth*layer_data_height*layer_data_width);
            for(int i=0;i<minibatch_size;i++) {
                reshapeThreeDMatrix2Col(P[M-1][i],i,X);
            }

            // Forward propagation
            threadController_master(fcnet_forward_prop_control_handle, THREAD_RESUME);
            
            threadController_master(fcnet_calc_loss_control_handle, THREAD_RESUME);

            float data_loss = losses[0][0];
            float reg_loss = losses[0][1];
            float accu = training_accuracy(Hs[fcnet_network_depth-1], network_params->fcnet_param->correct_labels);
            
            // Backward propagation
            threadController_master(fcnet_backward_prop_control_handle, THREAD_RESUME);
            //sleep(1);
            // Update weights
            threadController_master(fcnet_update_weights_control_handle, THREAD_RESUME);
            //sleep(1);

            if (verbose) {
                printf("CONVNET INFO: Epoch: %d iteration %d, data loss: %f, regulization loss: %f, total loss: %f, training accuracy: %f\n", e, iter, data_loss, reg_loss, data_loss + reg_loss,accu);
            }
            total_data_loss += losses[0][0];
            total_reg_loss += losses[0][1];
            training_accu += accu;
            //restoreThreeDMatrixFromCol(dP2D, dP3D);
            restoreThreeDMatrixFromCol(dP2D, dP[M-1]);
            // Backward propagation
            threadController_master(backward_prop_control_handle, THREAD_RESUME);
    
            // Update parameters
            threadController_master(update_weights_control_handle, THREAD_RESUME);
        }

        printf("CONVNET INFO:  Epoch: %d, data loss: %f, regulization loss: %f, total loss: %f, training accuracy: %f\n", e, total_data_loss/iterations, total_reg_loss/iterations, total_data_loss/iterations+total_reg_loss/iterations,training_accu/iterations);
        if (shuffle_training_samples != 0 && e % shuffle_training_samples == 0) {
            shuffleTrainingSamples(training_data, 
                network_params->fcnet_param->correct_labels,
                number_of_samples, 
                vertically_flip_training_samples, 
                horizontally_flip_training_samples,
                training_data,
                network_params->fcnet_param->correct_labels);
        }
        if (save_checkpoint != 0 && e != 0 && e % save_checkpoint == 0) {
            // Save checkpoints
            char checkpoint_counter[1000];
            sprintf(checkpoint_counter,"%d",e/save_checkpoint);
            int checkpoint_length = 20 + strlen(checkpoint_counter);
            char* checkpoint_filename = malloc(sizeof(char)*checkpoint_length);
            strcpy(checkpoint_filename,"checkpoint_");
            strcat(checkpoint_filename,checkpoint_counter);
            strcat(checkpoint_filename,".params");
            dumpConvnetConfig(M,N,
                filter_number,filter_stride_x, filter_stride_y, filter_width, filter_height, 
                enable_maxpooling,pooling_stride_x,pooling_stride_y,pooling_width,pooling_height,
                padding_width, padding_height,
                alpha, normalize_data_per_channel, K,
                F, b,
                Ws,bs,
                param_dir, checkpoint_filename);
        }
    }
    
    dumpConvnetConfig(M,N,
    filter_number,filter_stride_x, filter_stride_y, filter_width, filter_height, 
    enable_maxpooling,pooling_stride_x,pooling_stride_y,pooling_width,pooling_height,
    padding_width, padding_height,
    alpha, normalize_data_per_channel, K,
    F, b,
    Ws,bs,
    param_dir, network_params->params_filename);

    // For fun
    if (write_filters_as_images) {
        for(int i=0;i<M;i++) {
            for(int j=0;j<N;j++) {
                for(int k=0;k<filter_number[i*N+j];k++) {
                    char filter_name[100];
                    sprintf(filter_name,"F[%d][%d][%d]",i,j,k);
                    writeImage(F[i][j][k], filter_name, filter_image_dir);
                }
            }
        }
    }

    // Release memories, shutdown the network
    threadController_master(forward_prop_control_handle, THREAD_EXIT);
    threadController_master(backward_prop_control_handle, THREAD_EXIT);
    threadController_master(update_weights_control_handle, THREAD_EXIT);

    for(int i=0;i<M;i++) {
        for(int j=0;j<N;j++) {
            for(int k=0;k<filter_number[i*N+j];k++) {
                destroy3DMatrix(F[i][j][k]);
                destroy3DMatrix(dF[i][j][k]);
                destroy3DMatrix(b[i][j][k]);
                destroy3DMatrix(db[i][j][k]);
                if (use_rmsprop) {
                    destroy3DMatrix(Fcache[i][j][k]);
                    destroy3DMatrix(bcache[i][j][k]);
                }
            }
            for(int l=0;l<number_of_samples;l++) {
                destroy3DMatrix(C[i][j][l]);
                destroy3DMatrix(dC[i][j][l]);
            }
            free(F[i][j]);
            free(C[i][j]);
            free(b[i][j]);
            free(dF[i][j]);
            free(dC[i][j]);
            free(db[i][j]);
            if (use_rmsprop) {
                free(Fcache[i][j]);
                free(bcache[i][j]);
            }
        }
        free(F[i]);
        free(C[i]);
        free(b[i]);
        free(dF[i]);
        free(dC[i]);
        free(db[i]);
        if (use_rmsprop) {
            free(Fcache[i]);
            free(bcache[i]);
        }
        if (enable_maxpooling[i]) {
            for(int m=0;m<number_of_samples;m++) {
                destroy3DMatrix(P[i][m]);
                destroy3DMatrix(dP[i][m]);
            }
            free(P[i]);
            free(dP[i]);
        } else {
            P[i] = NULL;
            dP[i] = NULL;
        }
    }
    for(int i=0;i<K;i++) {
        destroy2DMatrix(Ws[i]);
        destroy2DMatrix(bs[i]);
    }
    free(Ws);
    free(bs);
    destroy2DMatrix(dP2D);
    for(int i=0;i<number_of_samples;i++) {
        destroy3DMatrix(dX[i]);
    }
    free(F);
    free(C);
    free(b);
    free(dF);
    free(dC);
    free(db);
    free(losses);
    destroy2DMatrix(X);
    if (use_rmsprop) {
        free(Fcache);
        free(bcache);
    }

    return 0;
}

int testConvnet_multithread(ConvnetParameters* convnet_params, TwoDMatrix* scores) {
    ThreeDMatrix**** F = NULL;
    ThreeDMatrix**** b = NULL;
    TwoDMatrix** Ws = NULL;
    TwoDMatrix** bs = NULL;
    int M = 0;
    int N = 0;
    int* filter_number = NULL;
    int* filter_stride_x = NULL;
    int* filter_stride_y = NULL;
    int* filter_width = NULL;
    int* filter_height = NULL;
    bool* enable_maxpooling = NULL;
    int* pooling_stride_x = NULL;
    int* pooling_stride_y = NULL;
    int* pooling_width = NULL;
    int* pooling_height = NULL;
    int* padding_width = NULL; 
    int* padding_height = NULL;
    float alpha = 0;
    bool normalize_data_per_channel = true;
    int K = 0;
    ThreeDMatrix** test_data = convnet_params->X;
    init2DMatrix(scores,convnet_params->number_of_samples,1);
    loadConvnetConfig(convnet_params->params_save_dir, convnet_params->params_filename,
        &M,&N,
        &filter_number,&filter_stride_x, &filter_stride_y, &filter_width, &filter_height, 
        &enable_maxpooling, &pooling_stride_x, &pooling_stride_y, &pooling_width, &pooling_height,
        &padding_width, &padding_height,
        &alpha, &normalize_data_per_channel, &K,
        &F,&b,
        &Ws,&bs);
    testConvnetCore( test_data, M, N, convnet_params->number_of_samples,
        filter_number, filter_stride_x,  filter_stride_y,  filter_width,  filter_height, 
        enable_maxpooling, pooling_stride_x, pooling_stride_y, pooling_width, pooling_height,
        padding_width,  padding_height,
        alpha, normalize_data_per_channel, K,
        F, b,
        Ws, bs,
        scores);
    return 0;
}

int testConvnetCore_multithread(ThreeDMatrix** test_data, int M,int N, int number_of_samples,
    int* filter_number,int* filter_stride_x, int* filter_stride_y, int* filter_width, int* filter_height, 
    bool* enable_maxpooling,int* pooling_stride_x,int* pooling_stride_y,int* pooling_width,int* pooling_height,
    int* padding_width, int* padding_height,
    float alpha, bool normalize_data_per_channel, int K,
    ThreeDMatrix**** F,ThreeDMatrix**** b,
    TwoDMatrix** Ws,TwoDMatrix** bs,
    TwoDMatrix* scores) {
    ThreeDMatrix**** C = malloc(sizeof(ThreeDMatrix***)*M);
    ThreeDMatrix*** P = malloc(sizeof(ThreeDMatrix**)*M);
    int layer_data_depth = test_data[0]->depth;
    int layer_data_height = test_data[0]->height;
    int layer_data_width = test_data[0]->width;
    for(int i=0;i<M;i++) {
        C[i] = (ThreeDMatrix***) malloc(sizeof(ThreeDMatrix**)*N);
        for(int j=0;j<N;j++) {
            C[i][j] = (ThreeDMatrix**) malloc(sizeof(ThreeDMatrix*)*number_of_samples);
            //int filter_depth = layer_data_depth;
            layer_data_depth = filter_number[i*N+j];
            layer_data_height = calcOutputSize(layer_data_height,padding_height[i*N+j],filter_height[i*N+j],filter_stride_y[i*N+j]);
            layer_data_width = calcOutputSize(layer_data_width,padding_width[i*N+j],filter_width[i*N+j],filter_stride_x[i*N+j]);
            for(int l=0;l<number_of_samples;l++) {
                C[i][j][l] = matrixMalloc(sizeof(ThreeDMatrix));
                init3DMatrix(C[i][j][l], layer_data_depth, layer_data_height, layer_data_width);
            }
        }
        if (enable_maxpooling[i]) {
            P[i] = (ThreeDMatrix**) malloc(sizeof(ThreeDMatrix*)*number_of_samples);
            layer_data_height = calcOutputSize(layer_data_height,0,pooling_height[i],pooling_stride_y[i]);
            layer_data_width = calcOutputSize(layer_data_width,0,pooling_width[i],pooling_stride_x[i]);
            for(int m=0;m<number_of_samples;m++) {
                P[i][m] = matrixMalloc(sizeof(ThreeDMatrix));
                init3DMatrix(P[i][m],layer_data_depth,layer_data_height,layer_data_width);
            }
        } else {
            P[i] = C[i][N-1];
        }
    }
    ThreeDMatrix** CONV_OUT = test_data;
    for(int i=0;i<M;i++) {
        for(int j=0;j<N;j++) {
            for(int n=0;n<number_of_samples;n++) {
                convLayerForward(CONV_OUT[n], 
                    F[i][j], 
                    filter_number[i*N+j], 
                    b[i][j], 
                    filter_height[i*N+j], 
                    filter_width[i*N+j], 
                    filter_stride_y[i*N+j], 
                    filter_stride_x[i*N+j], 
                    padding_height[i*N+j], 
                    padding_width[i*N+j], 
                    alpha, 
                    C[i][j][n]);
            }
            CONV_OUT = C[i][j];
        }
        if (enable_maxpooling[i]) {
            for(int n=0;n<number_of_samples;n++) {
                maxPoolingForward(CONV_OUT[n], 
                    pooling_stride_y[i], 
                    pooling_stride_x[i], 
                    pooling_width[i], 
                    pooling_height[i], 
                    P[i][n]);
            }
        } else {
            P[i] = CONV_OUT;
        }
        CONV_OUT = P[i];
    }
    TwoDMatrix* X = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(X,number_of_samples,layer_data_depth*layer_data_height*layer_data_width);
    for(int i=0;i<number_of_samples;i++) {
        reshapeThreeDMatrix2Col(P[M-1][i],i,X);
    }
    selftest(X,Ws,bs, alpha, K, 
        false, NULL, NULL, 0, NULL, NULL, 
        scores);
    return 0;
}

int CONV_forwardPropagation(int M, int N, int minibatch_size,ThreeDMatrix*** CONV_OUT_ADDR, ThreeDMatrix**** C, ThreeDMatrix*** P, ThreeDMatrix**** F, ThreeDMatrix**** b, int* filter_number, int* filter_height, int* filter_width, int* filter_stride_y, int* filter_stride_x, int* padding_width, int* padding_height, bool* enable_maxpooling, int* pooling_height, int* pooling_width, int* pooling_stride_x, int* pooling_stride_y, float alpha, bool verbose, int id, int number_of_threads) {
    int start_index = calc_h_start(id,minibatch_size,number_of_threads);
    int end_index = calc_h_end(id,minibatch_size,number_of_threads);
    ThreeDMatrix** CONV_OUT_local = *CONV_OUT_ADDR;
    for(int i=0;i<M;i++) {
        for(int j=0;j<N;j++) {
            #if defined(DEBUG) && DEBUG > 1
                pthread_mutex_lock(&debug_mutex);
                printf("DEBUG: CONV M = %d, N = %d, input matrix height: %d, width: %d, depth: %d\n", i, j,CONV_OUT_local[0]->height, CONV_OUT_local[0]->width, CONV_OUT_local[0]->depth);
                pthread_mutex_unlock(&debug_mutex);
            #endif
            //if (verbose) {
            //    printf("CONVNET INFO: Epoch: %d, CONV M = %d, N = %d\n", e, i, j);
            //}
            for(int n=start_index;n<=end_index;n++) {
                convLayerForward(CONV_OUT_local[n], 
                    F[i][j], 
                    filter_number[i*N+j], 
                    b[i][j], 
                    filter_height[i*N+j], 
                    filter_width[i*N+j], 
                    filter_stride_y[i*N+j], 
                    filter_stride_x[i*N+j], 
                    padding_height[i*N+j], 
                    padding_width[i*N+j], 
                    alpha, 
                    C[i][j][n]);
            }
            CONV_OUT_local = C[i][j];
            #if defined(DEBUG) && DEBUG > 1
                pthread_mutex_lock(&debug_mutex);
                printf("DEBUG: Output matrix height: %d, width: %d, depth: %d\n",CONV_OUT_local[0]->height, CONV_OUT_local[0]->width, CONV_OUT_local[0]->depth);
                pthread_mutex_unlock(&debug_mutex);
            #endif
        }
        // A thread barrier should be added here,
        // Or a local copy of CONV_OUT
        // Some thread may still trying to enter convLayerForward, while others have finished pooling and changed CONV_OUT
        // And then size error happens
        if (enable_maxpooling[i]) {
            //if (verbose) {
            //    printf("CONVNET INFO: Epoch: %d, POOLING M = %d\n", e, i);
            //}
            for(int n=start_index;n<=end_index;n++) {
                maxPoolingForward(CONV_OUT_local[n], 
                    pooling_stride_y[i], 
                    pooling_stride_x[i], 
                    pooling_width[i], 
                    pooling_height[i], 
                    P[i][n]);
            }
        } else {
            P[i] = CONV_OUT_local;
        }
        CONV_OUT_local = P[i];
    }
    return 0;
}

int CONV_backwardPropagation(int M, int N, int minibatch_size, ThreeDMatrix** training_data, ThreeDMatrix** dX, ThreeDMatrix**** C, ThreeDMatrix*** P, ThreeDMatrix**** F, ThreeDMatrix**** b, ThreeDMatrix**** dC, ThreeDMatrix*** dP, ThreeDMatrix**** dF, ThreeDMatrix**** db, int* filter_number, int* filter_height, int* filter_width, int* filter_stride_y, int* filter_stride_x, int* padding_width, int* padding_height, bool* enable_maxpooling, int* pooling_height, int* pooling_width, int* pooling_stride_x, int* pooling_stride_y, float alpha, bool verbose, int id,int number_of_threads) {
    int start_index = calc_h_start(id,minibatch_size,number_of_threads);
    int end_index = calc_h_end(id,minibatch_size,number_of_threads);
    for(int i=M-1;i>=0;i--) {
        if (enable_maxpooling[i]) {
            //if (verbose) {
            //    printf("CONVNET INFO: Epoch: %d, POOLING Backprop M = %d\n", e, i);
            //}
            for(int n=start_index;n<=end_index;n++) {
                maxPoolingBackward(dP[i][n], 
                    C[i][N-1][n], 
                    pooling_stride_y[i], 
                    pooling_stride_x[i], 
                    pooling_width[i], 
                    pooling_height[i],
                    dC[i][N-1][n]);
            }
        } else {
            // In cases where maxpooling is not enabled, dP is all zeros.
            for(int n=start_index;n<=end_index;n++) {
                dC[i][N-1][n] = dP[i][n];
            }
        }
    
        for(int j=N-1;j>=0;j--) {
            //if (verbose) {
            //    printf("CONVNET INFO: Epoch: %d, CONV Backprop M = %d, N = %d\n",e , i, j);
            //}
            if (i == 0 && j == 0) {
                /* This is the begining of the whole network
                ** So the input data should be training_data
                */ 
                for(int n=start_index;n<=end_index;n++) {
                    convLayerBackward(training_data[n], 
                        C[i][j][n],
                        F[i][j], 
                        dC[i][j][n], 
                        padding_height[i*N+j], 
                        padding_width[i*N+j], 
                        filter_stride_y[i*N+j], 
                        filter_stride_x[i*N+j],
                        alpha,
                        dX[n], 
                        dF[i][j], 
                        db[i][j]);
                }
            } else if (i != 0 && j == 0) {
                /* This is the begining of a CONV layer
                ** So the input data should be the output of the max pooling layer ahead of it, which is P[i-1][n]
                */
    
                for(int n=start_index;n<=end_index;n++) {
                    convLayerBackward(P[i-1][n], 
                        C[i][j][n],
                        F[i][j], 
                        dC[i][j][n], 
                        padding_height[i*N+j], 
                        padding_width[i*N+j],
                        filter_stride_y[i*N+j], 
                        filter_stride_x[i*N+j],
                        alpha,
                        dP[i-1][n], 
                        dF[i][j], 
                        db[i][j]);
                }
                //dV = dP[i-1];
            } else {
                for(int n=start_index;n<=end_index;n++) {
                    convLayerBackward(C[i][j-1][n], 
                        C[i][j][n],
                        F[i][j], 
                        dC[i][j][n], 
                        padding_height[i*N+j], 
                        padding_width[i*N+j],
                        filter_stride_y[i*N+j], 
                        filter_stride_x[i*N+j],
                        alpha,
                        dC[i][j-1][n], 
                        dF[i][j], 
                        db[i][j]);
                }
            }
        }
    }
    return 0;
}

int CONVNET_updateWeights(int M, int N, int minibatch_size, ThreeDMatrix**** C, ThreeDMatrix*** P, ThreeDMatrix**** F, ThreeDMatrix**** b, ThreeDMatrix**** dC, ThreeDMatrix*** dP, ThreeDMatrix**** dF, ThreeDMatrix**** db, ThreeDMatrix**** Fcache, ThreeDMatrix**** bcache, int* filter_number, float learning_rate, bool use_rmsprop, float rmsprop_decay_rate, float rmsprop_eps, int id, int number_of_threads) {
    if (use_rmsprop) {
        for(int i=0;i<M;i++) {
            for(int j=0;j<N;j++) {
                int start_index = calc_h_start(id,filter_number[i*N+j],number_of_threads);
                int end_index = calc_h_end(id,filter_number[i*N+j],number_of_threads);
                for(int k=start_index;k<=end_index;k++) {
                    RMSPropConvnet(F[i][j][k], dF[i][j][k], Fcache[i][j][k], learning_rate, rmsprop_decay_rate, rmsprop_eps, F[i][j][k]);
                    RMSPropConvnet(b[i][j][k], db[i][j][k], bcache[i][j][k], learning_rate, rmsprop_decay_rate, rmsprop_eps, b[i][j][k]);
                }
            }
        }
    } else {
        for(int i=0;i<M;i++) {
            for(int j=0;j<N;j++) {
                int start_index = calc_h_start(id,filter_number[i*N+j],number_of_threads);
                int end_index = calc_h_end(id,filter_number[i*N+j],number_of_threads);
                for(int k=start_index;k<=end_index;k++) {
                    vanillaUpdateConvnet(F[i][j][k], dF[i][j][k], learning_rate, F[i][j][k]);
                    vanillaUpdateConvnet(b[i][j][k], db[i][j][k], learning_rate, b[i][j][k]);
                }
            }
        }
    }
    return 0;
}

void* CONV_forwardPropagation_slave(void* args) {
    ConvnetSlaveArgs* a = args;
    int M = a->M;
    int N = a->N;
    int minibatch_size = a->minibatch_size;
    ThreeDMatrix*** CONV_OUT = a->CONV_OUT;
    //ThreeDMatrix** dX = a->dX;
    ThreeDMatrix**** C = a->C;
    ThreeDMatrix*** P = a->P;
    ThreeDMatrix**** F = a->F;
    ThreeDMatrix**** b = a->b;
    //ThreeDMatrix**** dC = a->dC;
    //ThreeDMatrix*** dP = a->dP;
    //ThreeDMatrix**** dF = a->dF;
    //ThreeDMatrix**** db = a->db;
    //ThreeDMatrix**** Fcache = a->Fcache;
    //ThreeDMatrix**** bcache = a->bcache;
    int* filter_number = a->filter_number;
    int* filter_height = a->filter_height;
    int* filter_width = a->filter_width;
    int* filter_stride_y = a->filter_stride_y;
    int* filter_stride_x = a->filter_stride_x;
    int* padding_width = a->padding_width;
    int* padding_height = a->padding_height;
    bool* enable_maxpooling = a->enable_maxpooling;
    int* pooling_height = a->pooling_height;
    int* pooling_width = a->pooling_width;
    int* pooling_stride_x = a->pooling_stride_x;
    int* pooling_stride_y = a->pooling_stride_y;
    float alpha = a->alpha;
    //bool use_rmsprop = a->use_rmsprop;
    //float learning_rate = a->learning_rate;
    int id = a->id;
    int number_of_threads = a->number_of_threads;
    bool verbose = a->verbose;
    ThreadControl* handle = a->handle;
    while(1) {
        threadController_slave(handle,CONTROL_WAIT_INST);
        #if defined(DEBUG) && DEBUG > 1
        pthread_mutex_lock(&debug_mutex);
        printf("DEBUG: Forward prop threads running\n");
        pthread_mutex_unlock(&debug_mutex);
        #endif
        CONV_forwardPropagation(M, N, minibatch_size,
        CONV_OUT, C, P, F, b, 
        filter_number, filter_height, filter_width, filter_stride_y, filter_stride_x,
        padding_width, padding_height, 
        enable_maxpooling, pooling_height, pooling_width, pooling_stride_x, pooling_stride_y, 
        alpha, verbose, id, number_of_threads);
        threadController_slave(handle,CONTROL_EXEC_COMPLETE);
    }
}

void* CONV_backwardPropagation_slave(void* args) {
    ConvnetSlaveArgs* a = args;
    int M = a->M;
    int N = a->N;
    int minibatch_size = a->minibatch_size;
    ThreeDMatrix** training_data = a->training_data;
    ThreeDMatrix** dX = a->dX;
    //ThreeDMatrix*** CONV_OUT = a->CONV_OUT;
    ThreeDMatrix**** C = a->C;
    ThreeDMatrix*** P = a->P;
    ThreeDMatrix**** F = a->F;
    ThreeDMatrix**** b = a->b;
    ThreeDMatrix**** dC = a->dC;
    ThreeDMatrix*** dP = a->dP;
    ThreeDMatrix**** dF = a->dF;
    ThreeDMatrix**** db = a->db;
    //ThreeDMatrix**** Fcache = a->Fcache;
    //ThreeDMatrix**** bcache = a->bcache;
    int* filter_number = a->filter_number;
    int* filter_height = a->filter_height;
    int* filter_width = a->filter_width;
    int* filter_stride_y = a->filter_stride_y;
    int* filter_stride_x = a->filter_stride_x;
    int* padding_width = a->padding_width;
    int* padding_height = a->padding_height;
    bool* enable_maxpooling = a->enable_maxpooling;
    int* pooling_height = a->pooling_height;
    int* pooling_width = a->pooling_width;
    int* pooling_stride_x = a->pooling_stride_x;
    int* pooling_stride_y = a->pooling_stride_y;
    float alpha = a->alpha;
    //bool use_rmsprop = a->use_rmsprop;
    //float learning_rate = a->learning_rate;
    int id = a->id;
    int number_of_threads = a->number_of_threads;
    bool verbose = a->verbose;
    ThreadControl* handle = a->handle;
    while(1) {
        threadController_slave(handle,CONTROL_WAIT_INST);
        #if defined(DEBUG) && DEBUG > 1
        pthread_mutex_lock(&debug_mutex);
        printf("DEBUG: Backward prop threads running\n");
        pthread_mutex_unlock(&debug_mutex);
        #endif
        CONV_backwardPropagation(M, N, minibatch_size, training_data, dX,
        C, P, F, b, dC, dP, dF, db, 
        filter_number, filter_height, filter_width, filter_stride_y, filter_stride_x, 
        padding_width, padding_height,
        enable_maxpooling, pooling_height, pooling_width, pooling_stride_x, pooling_stride_y,
        alpha, verbose, id, number_of_threads);
        threadController_slave(handle,CONTROL_EXEC_COMPLETE);
    }
}

void* CONV_updateWeights_slave(void* args) {
    ConvnetSlaveArgs* a = args;
    int M = a->M;
    int N = a->N;
    int minibatch_size = a->minibatch_size;
    //ThreeDMatrix*** CONV_OUT = a->CONV_OUT;
    ThreeDMatrix**** C = a->C;
    ThreeDMatrix*** P = a->P;
    ThreeDMatrix**** F = a->F;
    ThreeDMatrix**** b = a->b;
    ThreeDMatrix**** dC = a->dC;
    ThreeDMatrix*** dP = a->dP;
    ThreeDMatrix**** dF = a->dF;
    ThreeDMatrix**** db = a->db;
    ThreeDMatrix**** Fcache = a->Fcache;
    ThreeDMatrix**** bcache = a->bcache;
    int* filter_number = a->filter_number;
    //int* filter_height = a->filter_height;
    //int* filter_width = a->filter_width;
    //int* filter_stride_y = a->filter_stride_y;
    //int* filter_stride_x = a->filter_stride_x;
    //int* padding_width = a->padding_width;
    //int* padding_height = a->padding_height;
    //bool* enable_maxpooling = a->enable_maxpooling;
    //int* pooling_height = a->pooling_height;
    //int* pooling_width = a->pooling_width;
    //int* pooling_stride_x = a->pooling_stride_x;
    //int* pooling_stride_y = a->pooling_stride_y;
    //float alpha = a->alpha;
    bool use_rmsprop = a->use_rmsprop;
    float rmsprop_decay_rate = a->rmsprop_decay_rate;
    float rmsprop_eps = a->rmsprop_eps;
    float learning_rate = a->learning_rate;
    int id = a->id;
    int number_of_threads = a->number_of_threads;
    //bool verbose = a->verbose;
    ThreadControl* handle = a->handle;
    while(1) {
        threadController_slave(handle,CONTROL_WAIT_INST);
        #if defined(DEBUG) && DEBUG > 1
        pthread_mutex_lock(&debug_mutex);
        printf("DEBUG: Update weights threads running\n");
        pthread_mutex_unlock(&debug_mutex);
        #endif
        CONVNET_updateWeights(M, N, minibatch_size, 
        C, P, F, b, dC, dP, dF, db, Fcache, bcache, 
        filter_number, learning_rate, use_rmsprop, 
        rmsprop_decay_rate, rmsprop_eps, id, number_of_threads);
        threadController_slave(handle,CONTROL_EXEC_COMPLETE);
    }
}

void assignConvSlaveArguments (ConvnetSlaveArgs* args,
    int M, int N, int minibatch_size, ThreeDMatrix** training_data, ThreeDMatrix** dX, ThreeDMatrix*** CONV_OUT,
    ThreeDMatrix**** C, ThreeDMatrix*** P, ThreeDMatrix**** F, ThreeDMatrix**** b, ThreeDMatrix**** dC, ThreeDMatrix*** dP, ThreeDMatrix**** dF, ThreeDMatrix**** db, 
    ThreeDMatrix**** Fcache, ThreeDMatrix**** bcache,
    int* filter_number, int* filter_height, int* filter_width, int* filter_stride_y, int* filter_stride_x, 
    int* padding_width, int* padding_height, 
    bool* enable_maxpooling, int* pooling_height, int* pooling_width, int* pooling_stride_x, int* pooling_stride_y, 
    float learning_rate, bool use_rmsprop, float rmsprop_decay_rate, float rmsprop_eps,
    float alpha, bool verbose, int id,int number_of_threads, ThreadControl* handle) {
    args->M = M;
    args->N = N;
    args->minibatch_size = minibatch_size;
    args->training_data = training_data;
    args->CONV_OUT = CONV_OUT;
    args->dX = dX;
    args->C = C;
    args->P = P;
    args->F = F;
    args->b = b;
    args->dC = dC;
    args->dP = dP;
    args->dF = dF;
    args->db = db;
    args->Fcache = Fcache;
    args->bcache = bcache;
    args->filter_number = filter_number;
    args->filter_height = filter_height;
    args->filter_width = filter_width;
    args->filter_stride_y = filter_stride_y;
    args->filter_stride_x = filter_stride_x;
    args->padding_width = padding_width;
    args->padding_height = padding_height;
    args->enable_maxpooling = enable_maxpooling;
    args->pooling_height = pooling_height;
    args->pooling_width = pooling_width;
    args->pooling_stride_x = pooling_stride_x;
    args->pooling_stride_y = pooling_stride_y;
    args->learning_rate = learning_rate;
    args->use_rmsprop = use_rmsprop;
    args->rmsprop_decay_rate = rmsprop_decay_rate;
    args->rmsprop_eps = rmsprop_eps;
    args->alpha = alpha;
    args->verbose = verbose;
    args->id = id;
    args->number_of_threads = number_of_threads;
    args->handle = handle;
}
