#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <malloc.h>
#include <math.h>
#include <string.h>
#include "network_type.h"
#include "matrix_operations.h"
#include "layers.h"
#include "misc_utils.h"
#include "fully_connected_net.h"
#include "convnet_operations.h"
#include "convnet_layers.h"
#include "convnet.h"

int trainConvnet(ConvnetParameters* network_params) {
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
    // Turn these features off to reduce the complexity for now
    network_params->fcnet_param->use_momentum_update = false;
    network_params->fcnet_param->use_batchnorm = false;
    network_params->fcnet_param->use_nag_update = false;
    float current_fcnet_learning_rate = network_params->fcnet_param->learning_rate;
    network_params->fcnet_param->enable_learning_rate_step_decay = enable_learning_rate_step_decay;
    network_params->fcnet_param->enable_learning_rate_exponential_decay = enable_learning_rate_exponential_decay;
    network_params->fcnet_param->enable_learning_rate_invert_t_decay = enable_learning_rate_invert_t_decay;
    network_params->fcnet_param->learning_rate_decay_unit = learning_rate_decay_unit;
    network_params->fcnet_param->learning_rate_decay_a0 = learning_rate_decay_a0;
    network_params->fcnet_param->learning_rate_decay_k = learning_rate_decay_k;

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
    float* losses = malloc(sizeof(float)*3);
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
            ThreeDMatrix** CONV_OUT = training_data + iter*sizeof(ThreeDMatrix*);
            // Forward propagation
            for(int i=0;i<M;i++) {
                for(int j=0;j<N;j++) {
                    if (verbose) {
                        printf("CONVNET INFO: Epoch: %d, CONV M = %d, N = %d\n", e, i, j);
                    }
                    for(int n=0;n<minibatch_size;n++) {
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
                    //CONV_OUT = C[i][j];
                    #if defined(DEBUG) && DEBUG > 0
                    /**********************/
                    /******* DEBUG ********/
                    for(int n=0;n<minibatch_size;n++) {
                        printf("CONV M = %d, N = %d, INPUT %d\n",i,j,n);
                        print3DMatrix(CONV_OUT[n]);
                    }
                    for(int x=0;x<filter_number[i*N+j];x++) {
                        debugCheckingForNaNs3DMatrix(F[i][j][x], "after forward prop, F", x);
                        debugCheckingForNaNs3DMatrix(b[i][j][x], "after forward prop, b", x);
                        printf("F[%d][%d][%d]\n", i, j, x);
                        print3DMatrix(F[i][j][x]);
                        printf("b[%d][%d][%d]\n", i, j, x);
                        print3DMatrix(b[i][j][x]);
                    }
                    for(int n=0;n<minibatch_size;n++) {
                        debugCheckingForNaNs3DMatrix(C[i][j][n], "after forward prop, C", n);
                        printf("C[%d][%d][%d]\n", i, j, n);
                        print3DMatrix(C[i][j][n]);
                        printf("0x0: CONV M = %d, N = %d, C = %d, %f\n",i,j,n,C[i][j][n]->d[0][0][0]);
                        printf("1x1: CONV M = %d, N = %d, C = %d, %f\n",i,j,n,C[i][j][n]->d[0][1][1]);
                    }
                    /******* DEBUG ********/
                    /**********************/
                    #endif
                    CONV_OUT = C[i][j];
                }
                if (enable_maxpooling[i]) {
                    if (verbose) {
                        printf("CONVNET INFO: Epoch: %d, POOLING M = %d\n", e, i);
                    }
                    for(int n=0;n<minibatch_size;n++) {
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
    
                #if defined(DEBUG) && DEBUG > 0
                /**********************/
                /******* DEBUG ********/
                for(int n=0;n<minibatch_size;n++) {
                    printf("P[%d][%d]\n", i, n);
                    print3DMatrix(P[i][n]);
                    debugCheckingForNaNs3DMatrix(P[i][n], "after forward prop, P", n);
                }
                /******* DEBUG ********/
                /**********************/
                #endif
            }
    
            // Feed data to fully connected network
            TwoDMatrix* X = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrix(X,minibatch_size,layer_data_depth*layer_data_height*layer_data_width);
            for(int i=0;i<minibatch_size;i++) {
                reshapeThreeDMatrix2Col(P[M-1][i],i,X);
            }
    
            #if defined(DEBUG) && DEBUG > 0
            /**********************/
            /******* DEBUG ********/
            debugCheckingForNaNs2DMatrix(X, "after CONV->FC reshape, X", 0);
            printf("fully connected net input:\n");
            printMatrix(X);
            /******* DEBUG ********/
            /**********************/
            #endif
    
            network_params->fcnet_param->X = X;
    
            FCTrainCore(network_params->fcnet_param, 
                Ws, bs, 
                NULL, NULL, NULL, NULL,
                Wscache, bscache,
                NULL, NULL, NULL, NULL,
                dP2D, e, &current_fcnet_learning_rate, losses);
            destroy2DMatrix(X);
            #if defined(DEBUG) && DEBUG > 0
            /**********************/
            /******* DEBUG ********/
            printf("dP2D\n");
            printMatrix(dP2D);
            debugCheckingForNaNs2DMatrix(dP2D, "after FC back prop, dP2D", 0);
            /******* DEBUG ********/
            /**********************/
            #endif
            if (verbose) {
                printf("CONVNET INFO: Epoch: %d iteration %d, data loss: %f, regulization loss: %f, total loss: %f, training accuracy: %f\n", e, iter, losses[0], losses[1], losses[0]+losses[1],losses[2]);
            }
            total_data_loss += losses[0];
            total_reg_loss += losses[1];
            training_accu += losses[2];
            //restoreThreeDMatrixFromCol(dP2D, dP3D);
            restoreThreeDMatrixFromCol(dP2D, dP[M-1]);
            #if defined(DEBUG) && DEBUG > 0
            /**********************/
            /******* DEBUG ********/
            for(int n=0;n<minibatch_size;n++) {
                debugCheckingForNaNs3DMatrix(dP[M-1][n], "after FC->CONV reshape, dP[M-1][%d]", n);
                //printf("dP[M-1][%d]\n", n);
                //print3DMatrix(dP[M-1][n]);
            }
            /******* DEBUG ********/
            /**********************/
            #endif
    
            // Backward propagation
            /* Schematic
            ** For the (i*j)th layer of a M*N layer network
            **
            **             P[i-1][j]  F[i][j],b[i][j]  C[i][j]                 P[i][j]
            **             |                 |         |                         |
            **             V                 V         V                         V
            ** dF[i][j]    -----------------------------                -------------------
            ** db[i][j] <- |         CONV[i][j]        | <- dC[i][j] <- |  POOLING[i][j]  | <- dP[i][j]
            ** dP[i][j]    -----------------------------                -------------------
            **
            ** Actual dimentions of each variable:
            ** dC[M][N][minibatch_size]
            ** dP[M][minibatch_size]
            ** dF[M][N][filter_number]
            ** db[M][N][filter_number]
            **
            */
            //ThreeDMatrix** dV = dP3D;
            //dP[M-1] = dP3D;
            for(int i=M-1;i>=0;i--) {
                if (enable_maxpooling[i]) {
                    if (verbose) {
                        printf("CONVNET INFO: Epoch: %d, POOLING Backprop M = %d\n", e, i);
                    }
                    for(int n=0;n<minibatch_size;n++) {
                        #if defined(DEBUG) && DEBUG > 0
                        printf("dP[%d][%d]\n", i,n);
                        print3DMatrix(dP[i][n]);
                        #endif
                        maxPoolingBackward(dP[i][n], 
                            C[i][N-1][n], 
                            pooling_stride_y[i], 
                            pooling_stride_x[i], 
                            pooling_width[i], 
                            pooling_height[i],
                            dC[i][N-1][n]);
                        #if defined(DEBUG) && DEBUG > 0
                        printf("dC[%d][N-1][%d]\n", i,n);
                        print3DMatrix(dC[i][N-1][n]);
                        #endif
                    }
                } else {
                    // FIX ME
                    // In cases where maxpooling is not enabled, dP is all zeros.
                    for(int n=0;n<minibatch_size;n++) {
                        dC[i][N-1][n] = dP[i][n];
                    }
                }
    
                #if defined(DEBUG) && DEBUG > 0
                /**********************/
                /******* DEBUG ********/
                for(int n=0;n<minibatch_size;n++) debugCheckingForNaNs3DMatrix(dC[i][N-1][n], "after max pooling backprop, dC", n);
                /******* DEBUG ********/
                /**********************/
                #endif
    
                for(int j=N-1;j>=0;j--) {
                    if (verbose) {
                        printf("CONVNET INFO: Epoch: %d, CONV Backprop M = %d, N = %d\n",e , i, j);
                    }
                    if (i == 0 && j == 0) {
                        /* This is the begining of the whole network
                        ** So the input data should be training_data
                        */ 
    
                        #if defined(DEBUG) && DEBUG > 0
                        /*****************/
                        /***** DEBUG *****/
                        printf("This is the beginning of a network\n");
                        /***** DEBUG *****/
                        /*****************/
                        #endif
                        for(int n=0;n<minibatch_size;n++) {
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
                        #if defined(DEBUG) && DEBUG > 0
                        /*****************/
                        /***** DEBUG *****/
                        printf("This is the beginning of a conv layer\n");
                        /***** DEBUG *****/
                        /*****************/
                        #endif
    
                        for(int n=0;n<minibatch_size;n++) {
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
                        for(int n=0;n<minibatch_size;n++) {
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
    
                    #if defined(DEBUG) && DEBUG > 0
                    /**********************/
                    /******* DEBUG ********/
                    for(int x=0;x<dC[i][j][0]->depth;x++) {
                        debugCheckingForNaNs3DMatrix(dF[i][j][x], "after backprop, dF", x);
                        debugCheckingForNaNs3DMatrix(db[i][j][x], "after backprop, db", x);
                        printf("dF[%d][%d][%d]\n",i,j,x);
                        print3DMatrix(dF[i][j][x]);
                        printf("db[%d][%d][%d]\n",i,j,x);
                        print3DMatrix(db[i][j][x]);
                    }
                    for(int n=0;n<minibatch_size;n++) debugCheckingForNaNs3DMatrix(dC[i][j][n], "after backprop, dC", n);
    
                    /******* DEBUG ********/
                    /**********************/
                    #endif
                }
            }
    
            // Update parameters
            if (use_rmsprop) {
                for(int i=0;i<M;i++) {
                    for(int j=0;j<N;j++) {
                        for(int k=0;k<filter_number[i*N+j];k++) {
                            RMSPropConvnet(F[i][j][k], dF[i][j][k], Fcache[i][j][k], learning_rate, rmsprop_decay_rate, rmsprop_eps, F[i][j][k]);
                            RMSPropConvnet(b[i][j][k], db[i][j][k], bcache[i][j][k], learning_rate, rmsprop_decay_rate, rmsprop_eps, b[i][j][k]);
                        }
                    }
                }
            } else {
                for(int i=0;i<M;i++) {
                    for(int j=0;j<N;j++) {
                        for(int k=0;k<filter_number[i*N+j];k++) {
                            vanillaUpdateConvnet(F[i][j][k], dF[i][j][k], learning_rate, F[i][j][k]);
                            vanillaUpdateConvnet(b[i][j][k], db[i][j][k], learning_rate, b[i][j][k]);
                        }
                    }
                }
            }
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
    if (use_rmsprop) {
        free(Fcache);
        free(bcache);
    }

    return 0;
}

int testConvnet(ConvnetParameters* convnet_params, TwoDMatrix* scores) {
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

int testConvnetCore(ThreeDMatrix** test_data, int M,int N, int number_of_samples,
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
