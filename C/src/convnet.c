#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <malloc.h>
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
    bool enable_padding = network_params->enable_padding;
    int padding_width = network_params->padding_width;
    int padding_height = network_params->padding_height;
    float alpha = network_params->alpha;
    float learning_rate = network_params->learning_rate;
    float base_learning_rate = learning_rate;
    bool verbose = network_params->verbose;

    bool normalize_data_per_channel = network_params->normalize_data_per_channel;
    
    bool enable_learning_rate_step_decay = network_params->enable_learning_rate_step_decay;
    bool enable_learning_rate_exponential_decay = network_params->enable_learning_rate_exponential_decay;
    bool enable_learning_rate_invert_t_decay = network_params->enable_learning_rate_invert_t_decay;
    int learning_rate_decay_unit = network_params->learning_rate_decay_unit;
    float learning_rate_decay_a0 = network_params->learning_rate_decay_a0;
    float learning_rate_decay_k = network_params->learning_rate_decay_k;
    // Turn these features off to reduce the complexity for now
    network_params->fcnet_param->use_momentum_update = false;
    network_params->fcnet_param->use_batchnorm = false;
    network_params->fcnet_param->use_nag_update = false;
    network_params->fcnet_param->use_rmsprop = false;
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

    if (enable_padding) {
        for(int i=0;i<number_of_samples;i++) {
            ThreeDMatrix* tmp = matrixMalloc(sizeof(ThreeDMatrix));
            zeroPadding(training_data[i],padding_height,padding_width,tmp);
            destroy3DMatrix(training_data[i]);
            training_data[i] = tmp;
        }
    }

    printf("CONVNET INFO: Initializing learnable weights and intermediate layers\n");
    unsigned long long int total_parameters = 0;
    unsigned long long int total_memory = 0;
    float* losses = malloc(sizeof(float)*2);
    /*
    C will hold intermediate values of CONV -> RELU layer, C[M][N][number_of_samples]
    P will hold intermediate values of POOL, P[M][number_of_samples]
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
    TwoDMatrix* dP2D = matrixMalloc(sizeof(TwoDMatrix));
    ThreeDMatrix** dP3D = malloc(sizeof(ThreeDMatrix*)*number_of_samples);

    ThreeDMatrix** dX = malloc(sizeof(ThreeDMatrix*)*number_of_samples);
    for(int i=0;i<number_of_samples;i++) {
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
        for(int j=0;j<N;j++) {
            F[i][j] = (ThreeDMatrix**) malloc(sizeof(ThreeDMatrix*)*filter_number[i*M+j]);
            b[i][j] = (ThreeDMatrix**) malloc(sizeof(ThreeDMatrix*)*filter_number[i*M+j]);
            C[i][j] = (ThreeDMatrix**) malloc(sizeof(ThreeDMatrix*)*number_of_samples);
            dF[i][j] = (ThreeDMatrix**) malloc(sizeof(ThreeDMatrix*)*filter_number[i*M+j]);
            db[i][j] = (ThreeDMatrix**) malloc(sizeof(ThreeDMatrix*)*filter_number[i*M+j]);
            dC[i][j] = (ThreeDMatrix**) malloc(sizeof(ThreeDMatrix*)*number_of_samples);
            for(int k=0;k<filter_number[i*M+j];k++) {
                F[i][j][k] = matrixMalloc(sizeof(ThreeDMatrix));
                b[i][j][k] = matrixMalloc(sizeof(ThreeDMatrix));
                dF[i][j][k] = matrixMalloc(sizeof(ThreeDMatrix));
                db[i][j][k] = matrixMalloc(sizeof(ThreeDMatrix));
                init3DMatrixNormRand(F[i][j][k],layer_data_depth,filter_height[i*M+j],filter_width[i*M+j],0.0,1.0);
                init3DMatrix(b[i][j][k],1,1,1);
                init3DMatrix(dF[i][j][k],layer_data_depth,filter_height[i*M+j],filter_width[i*M+j]);
                init3DMatrix(db[i][j][k],1,1,1);
            }
            int filter_depth = layer_data_depth;
            layer_data_depth = filter_number[i*M+j];
            layer_data_height = calcOutputSize(layer_data_height,0,filter_height[i*M+j],filter_stride_y[i*M+j]);
            layer_data_width = calcOutputSize(layer_data_width,0,filter_width[i*M+j],filter_stride_x[i*M+j]);
            for(int l=0;l<number_of_samples;l++) {
                C[i][j][l] = matrixMalloc(sizeof(ThreeDMatrix));
                dC[i][j][l] = matrixMalloc(sizeof(ThreeDMatrix));
                init3DMatrix(C[i][j][l], layer_data_depth, layer_data_height, layer_data_width);
                init3DMatrix(dC[i][j][l], layer_data_depth, layer_data_height, layer_data_width);
            }
            printf("CONVNET INFO: CONV[%dx%d-%d]: [%dx%dx%d]\t\tweights: (%d*%d*%d)*%d=%d\n",
                filter_width[i*M+j],filter_height[i*M+j],filter_depth, layer_data_width,layer_data_height,layer_data_depth,
                filter_width[i*M+j],filter_height[i*M+j],filter_depth,layer_data_depth,filter_width[i*M+j]*filter_height[i*M+j]*filter_depth*layer_data_depth);
            total_memory += layer_data_depth*layer_data_height*layer_data_width;
            total_parameters += filter_width[i*M+j]*filter_height[i*M+j]*filter_depth*layer_data_depth;
        }
        if (enable_maxpooling[i]) {
            P[i] = (ThreeDMatrix**) malloc(sizeof(ThreeDMatrix*)*number_of_samples);
            dP[i] = (ThreeDMatrix**) malloc(sizeof(ThreeDMatrix*)*number_of_samples);
            layer_data_height = calcOutputSize(layer_data_height,0,pooling_height[i],pooling_stride_y[i]);
            layer_data_width = calcOutputSize(layer_data_width,0,pooling_width[i],pooling_stride_x[i]);
            for(int m=0;m<number_of_samples;m++) {
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
    for(int i=0;i<number_of_samples;i++) {
        dP3D[i] = matrixMalloc(sizeof(ThreeDMatrix));
        init3DMatrix(dP3D[i],P[M-1][i]->depth, P[M-1][i]->height, P[M-1][i]->width);
    }
    // Initialize the fully connected network in convnet
    int* fcnet_hidden_layer_sizes = network_params->fcnet_param->hidden_layer_sizes;
    int K = network_params->fcnet_param->network_depth;
    TwoDMatrix** Ws = malloc(sizeof(TwoDMatrix*)*K);
    TwoDMatrix** bs = malloc(sizeof(TwoDMatrix*)*K);
    int fcnet_labels = network_params->fcnet_param->labels;
    fcnet_hidden_layer_sizes[K-1] = fcnet_labels;
    printf("FCNET INFO: INPUT[%dx%d]\t\t\tweights: 0\n",number_of_samples,layer_data_depth*layer_data_height*layer_data_width);
    total_memory += layer_data_depth*layer_data_height*layer_data_width;
    int former_width = layer_data_depth*layer_data_height*layer_data_width;
    for(int i=0;i<K;i++) {
        // Initialize layer data holders
        Ws[i] = matrixMalloc(sizeof(TwoDMatrix));
        bs[i] = matrixMalloc(sizeof(TwoDMatrix));
        init2DMatrixNormRand(Ws[i],former_width,fcnet_hidden_layer_sizes[i],0.0,1.0, former_width);
        init2DMatrixZero(bs[i],1,fcnet_hidden_layer_sizes[i]);
        printf("FCNET INFO: FC[%dx%dx%d]\t\t\tweights: %d*%d=%d\n",1,1,fcnet_hidden_layer_sizes[i],former_width,fcnet_hidden_layer_sizes[i],former_width*fcnet_hidden_layer_sizes[i]);
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
    network_params->fcnet_param->minibatch_size = number_of_samples;
    network_params->fcnet_param->epochs = 1;
    // Print some statistical info
    printf("CONVNET INFO: Total parameters: %lld\n",total_parameters);
    char memory_unit_per_image = determineMemoryUnit(total_memory*sizeof(float));
    float memory_usage_per_image = memoryUsageReadable(total_memory*sizeof(float),memory_unit_per_image);
    char memory_unit_total = determineMemoryUnit(total_memory*sizeof(float)*number_of_samples);
    float memory_usage_total = memoryUsageReadable(total_memory*sizeof(float)*number_of_samples,memory_unit_total);
    printf("CONVNET INFO: Memory usage: %f%cB per image, total memory: %f%cB\n",memory_usage_per_image, memory_unit_per_image, memory_usage_total, memory_unit_total);
    
    // Start training the network
    /*
    C will hold intermediate values of CONV -> RELU layer, C[M][N][number_of_samples]
    P will hold intermediate values of POOL, P[M][number_of_samples]
    F will be a 2D array that contains filters, F[M][N][filter_number]
    b will be a 2D array that holds biases, b[M][N][filter_number]
    */
    /* INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC */
    ThreeDMatrix** CONV_OUT = training_data;
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
        // Forward propagation
        for(int i=0;i<M;i++) {
            for(int j=0;j<N;j++) {
                if (verbose) {
                    printf("CONVNET INFO: Epoch: %d, CONV M = %d, N = %d\n", e, i, j);
                }
                for(int n=0;n<number_of_samples;n++) {
                    convLayerForward(CONV_OUT[n], 
                        F[i][j], 
                        filter_number[i*M+j], 
                        b[i][j], 
                        filter_height[i*M+j], 
                        filter_width[i*M+j], 
                        filter_stride_y[i*M+j], 
                        filter_stride_x[i*M+j], 
                        0, 
                        0, 
                        alpha, 
                        C[i][j][n]);
                }
                CONV_OUT = C[i][j];
                
                /**********************/
                /******* DEBUG ********/
                for(int x=0;x<filter_number[i*M+j];x++) {
                    debugCheckingForNaNs3DMatrix(F[i][j][x], "after forward prop, F", x);
                    debugCheckingForNaNs3DMatrix(b[i][j][x], "after forward prop, b", x);
                }
                for(int n=0;n<number_of_samples;n++) debugCheckingForNaNs3DMatrix(C[i][j][n], "after forward prop, C", n);
                /******* DEBUG ********/
                /**********************/

            }
            if (enable_maxpooling[i]) {
                if (verbose) {
                    printf("CONVNET INFO: Epoch: %d, POOLING M = %d\n", e, i);
                }
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
            
            /**********************/
            /******* DEBUG ********/
            for(int n=0;n<number_of_samples;n++) debugCheckingForNaNs3DMatrix(P[i][n], "after forward prop, P", n);
            /******* DEBUG ********/
            /**********************/

        }

        // Feed data to fully connected network
        TwoDMatrix* X = matrixMalloc(sizeof(TwoDMatrix));
        init2DMatrix(X,number_of_samples,layer_data_depth*layer_data_height*layer_data_width);
        for(int i=0;i<number_of_samples;i++) {
            reshapeThreeDMatrix2Col(P[M-1][i],i,X);

        }

        /**********************/
        /******* DEBUG ********/
        debugCheckingForNaNs2DMatrix(X, "after CONV->FC reshape, X", 0);
        /******* DEBUG ********/
        /**********************/

        network_params->fcnet_param->X = X;

        FCTrainCore(network_params->fcnet_param, 
            Ws, bs, 
            NULL, NULL, NULL, NULL,
            NULL, NULL,
            NULL, NULL, NULL, NULL,
            dP2D, e, &current_fcnet_learning_rate, losses);

        /**********************/
        /******* DEBUG ********/
        //printf("dP2D\n");
        //printMatrix(dP2D);
        debugCheckingForNaNs2DMatrix(dP2D, "after FC back prop, dP2D", 0);
        /******* DEBUG ********/
        /**********************/
        if (e % 1000 == 0 || verbose) {
            printf("CONVNET INFO: Epoch: %d, data loss: %f, regulization loss: %f, total loss: %f\n", e, losses[0], losses[1], losses[0]+losses[1]);
        }
        restoreThreeDMatrixFromCol(dP2D, dP3D);
        /**********************/
        /******* DEBUG ********/
        for(int n=0;n<number_of_samples;n++) {
            debugCheckingForNaNs3DMatrix(dP3D[n], "after FC->CONV reshape, dP3D", n);
            //printf("dP3D[%d]\n", n);
            //print3DMatrix(dP3D[n]);
        }
        /******* DEBUG ********/
        /**********************/
        
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
        ** dC[M][N][number_of_samples]
        ** dP[M][number_of_samples]
        ** dF[M][N][filter_number]
        ** db[M][N][filter_number]
        **
        */
        ThreeDMatrix** dV = dP3D;
        for(int i=M-1;i>=0;i--) {
            if (enable_maxpooling[i]) {
                if (verbose) {
                    printf("CONVNET INFO: Epoch: %d, POOLING Backprop M = %d\n", e, i);
                }
                for(int n=0;n<number_of_samples;n++) {
                    maxPoolingBackward(dV[n], 
                        P[i][n], 
                        pooling_stride_y[i], 
                        pooling_stride_x[i], 
                        pooling_width[i], 
                        pooling_height[i],
                        dC[i][N-1][n]);
                }
            } else {
                for(int n=0;n<number_of_samples;n++) {
                    dC[i][N-1][n] = dP[i][n];
                }
            }

            /**********************/
            /******* DEBUG ********/
            for(int n=0;n<number_of_samples;n++) debugCheckingForNaNs3DMatrix(dC[i][N-1][n], "after max pooling backprop, dC", n);
            /******* DEBUG ********/
            /**********************/

            for(int j=N-1;j>=0;j--) {
                if (verbose) {
                    printf("CONVNET INFO: Epoch: %d, CONV Backprop M = %d, N = %d\n",e , i, j);
                }
                if (i == 0 && j == 0) {
                    /* This is the begining of the whole network
                    ** So the input data should be training_data
                    */ 
                    for(int n=0;n<number_of_samples;n++) {
                        convLayerBackward(training_data[n], 
                            C[i][j][n],
                            F[i][j], 
                            dC[i][j][n], 
                            0, 
                            0, 
                            filter_stride_y[i*M+j], 
                            filter_stride_x[i*M+j],
                            alpha,
                            dX[n], 
                            dF[i][j], 
                            db[i][j]);
                    }
                } else if (i != 0 && j == 0) {
                    /* This is the begining of a CONV layer
                    ** So the input data should be the output of the max pooling layer ahead of it, which is P[i-1][n]
                    */
                    for(int n=0;n<number_of_samples;n++) {
                        convLayerBackward(P[i-1][n], 
                            C[i][j][n],
                            F[i][j], 
                            dC[i][j][n], 
                            0, 
                            0, 
                            filter_stride_y[i*M+j], 
                            filter_stride_x[i*M+j],
                            alpha,
                            dP[i-1][n], 
                            dF[i][j], 
                            db[i][j]);
                    }
                    dV = dP[i-1];
                } else {
                    for(int n=0;n<number_of_samples;n++) {
                        convLayerBackward(C[i][j-1][n], 
                            C[i][j][n],
                            F[i][j], 
                            dC[i][j][n], 
                            0, 
                            0, 
                            filter_stride_y[i*M+j], 
                            filter_stride_x[i*M+j],
                            alpha,
                            dC[i][j-1][n], 
                            dF[i][j], 
                            db[i][j]);
                    }
                }

                /**********************/
                /******* DEBUG ********/
                for(int x=0;x<dC[i][j][0]->depth;x++) {
                    debugCheckingForNaNs3DMatrix(dF[i][j][x], "after backprop, dF", x);
                    debugCheckingForNaNs3DMatrix(db[i][j][x], "after backprop, db", x);
                }
                for(int n=0;n<number_of_samples;n++) debugCheckingForNaNs3DMatrix(dC[i][j][n], "after backprop, dC", n);
                /******* DEBUG ********/
                /**********************/
                
            }
        }
        
        // Update parameters
        for(int i=0;i<M;i++) {
            for(int j=0;j<N;j++) {
                for(int k=0;k<filter_number[i*M+j];k++) {
                    vanillaUpdateConvnet(F[i][j][k], dF[i][j][k], learning_rate, F[i][j][k]);
                    vanillaUpdateConvnet(b[i][j][k], db[i][j][k], learning_rate, b[i][j][k]);
                }
            }
        }
    }
    

    // Release memories, shutdown the network
    for(int i=0;i<M;i++) {
        for(int j=0;j<N;j++) {
            for(int k=0;k<filter_number[i*M+j];k++) {
                destroy3DMatrix(F[i][j][k]);
                destroy3DMatrix(dF[i][j][k]);
                destroy3DMatrix(b[i][j][k]);
                destroy3DMatrix(db[i][j][k]);
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
        }
        free(F[i]);
        free(C[i]);
        free(b[i]);
        free(dF[i]);
        free(dC[i]);
        free(db[i]);
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
        destroy3DMatrix(dP3D[i]);
    }
    free(dP3D);
    free(dX);
    
    return 0;
}
