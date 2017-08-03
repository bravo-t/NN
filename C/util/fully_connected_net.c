#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <stdarg.h>
#include <stdbool.h>
#include "matrix_operations.h"
#include "misc_utils.h"
#include "layers.h"
#include "fully_connected_net.h"

//int fullyConnectedNets(TwoDMatrix* X, int minibatch_size) {
//
//}

parameters* initTrainParameters(TwoDMatrix* X, 
    TwoDMatrix* correct_labels, 
    int minibatch_size, 
    int labels, 
    float learning_rate, 
    float reg_strength, 
    float alpha, 
    int epochs,
    int network_depth, ...) {
    va_list size_configs;
    int* layer_sizes = malloc(sizeof(int)*network_depth);
    va_start(size_configs,network_depth);
    for(int i=0;i<network_depth;i++) {
        layer_sizes[i] = va_arg(size_configs,int);
    }
    va_end(size_configs);
    // The last layer is the label layer, so you don't have control on the size of it
    layer_sizes[network_depth-1] = labels;

    printf("INFO: Setting up basic parameters for the network\n");
    printf("INFO: Sizes of networks are: ");
    for(int i=0;i<network_depth;i++) {
        printf("%d ",layer_sizes[i]);
    }
    printf("\n");
    parameters* network_params = malloc(sizeof(parameters));
    if(network_params == NULL) {
        printf("ERROR: Cannot allocate memory for parameters, exiting...\n");
        exit(1);
    }
    network_params->X = X;
    network_params->correct_labels = correct_labels;
    network_params->minibatch_size = minibatch_size;
    network_params->labels = labels;
    network_params->reg_strength = reg_strength;
    network_params->alpha = alpha;
    network_params->learning_rate = learning_rate;
    network_params->epochs = epochs;
    network_params->network_depth = network_depth;
    network_params->hidden_layer_sizes = layer_sizes;
    return network_params;
}

/*
 How will the size of the Ws and Hs determined?
 Like below (X is the input data):
 int former_width = X->width;
 TwoDMatrix** Ws;
 TwoDMatrix** bs;
 for(int i=0;i<network_depth;i++) {
    init2DMatrix(Ws[i],former_width,hidden_layer_sizes[i]);
    init2DMatrix(bs[i],1,hidden_layer_sizes[i]);
    former_width = hidden_layer_sizes[i];
 }
 */
int train(parameters* network_params) {
    TwoDMatrix* training_data = network_params->X;
    TwoDMatrix* correct_labels = network_params->correct_labels;
    int minibatch_size = network_params->minibatch_size;
    int labels = network_params->labels;
    float reg_strength = network_params->reg_strength;
    float alpha = network_params->alpha;
    float learning_rate = network_params->learning_rate;
    int network_depth = network_params->network_depth;
    int* hidden_layer_sizes = network_params->hidden_layer_sizes;
    int epochs = network_params->epochs;
    // Below are control variables for optimizers
    bool use_momentum_update = true;
    bool use_nag_update = false;
    bool use_rmsprop = false;
    float mu = 0.5f; // or 0.5,0.95, 0.99
    float decay_rate = 0.99f; // or with more 9s in it
    float eps = 1e-6;
    // Initialize all learnable parameters
    printf("INFO: Initializing all required learnable parameters for the network\n");
    int number_of_weights = 0;
    int size_of_Ws = 0;
    int number_of_biases = 0;
    int size_of_bs = 0;
    int number_of_hvalues = 0;
    int size_of_Hs = 0;
    int size_of_a_matrix = sizeof(TwoDMatrix);
    // Weights
    TwoDMatrix** Ws = malloc(sizeof(TwoDMatrix*)*network_depth);
    // Biases
    TwoDMatrix** bs = malloc(sizeof(TwoDMatrix*)*network_depth);
    // Hidden layers
    TwoDMatrix** Hs = malloc(sizeof(TwoDMatrix*)*network_depth);
    // Gradient descend values of Weights
    TwoDMatrix** dWs = malloc(sizeof(TwoDMatrix*)*network_depth);
    // Gradient descend values of Biases
    TwoDMatrix** dbs = malloc(sizeof(TwoDMatrix*)*network_depth);
    // Gradient descend values of Hidden layers
    TwoDMatrix** dHs = malloc(sizeof(TwoDMatrix*)*network_depth);
    // Below variables are used in optimization algorithms
    TwoDMatrix** vWs = NULL;
    TwoDMatrix** vW_prevs = NULL;
    TwoDMatrix** vbs = NULL;
    TwoDMatrix** vb_prevs = NULL;
    TwoDMatrix** Wcaches = NULL;
    TwoDMatrix** bcaches = NULL;

    if (use_momentum_update) {
        printf("INFO: Momentum update is used\n");
        vWs = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
        vbs = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
    }
    if (use_nag_update) {
        printf("INFO: NAG update is used\n");
        vWs = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
        vbs = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
        vW_prevs = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
        vb_prevs = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
    }
    if (use_rmsprop) {
        printf("INFO: RMSProp is used\n");
        Wcaches = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
        bcaches = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
    }

    int former_width = training_data->width;
    for(int i=0;i<network_depth;i++) {
        // Initialize layer data holders
        Ws[i] = matrixMalloc(sizeof(TwoDMatrix));
        bs[i] = matrixMalloc(sizeof(TwoDMatrix));
        Hs[i] = matrixMalloc(sizeof(TwoDMatrix));
        dWs[i] = matrixMalloc(sizeof(TwoDMatrix));
        dbs[i] = matrixMalloc(sizeof(TwoDMatrix));
        dHs[i] = matrixMalloc(sizeof(TwoDMatrix));
        init2DMatrixNormRand(Ws[i],former_width,hidden_layer_sizes[i],0.0,1.0);
        init2DMatrixZero(bs[i],1,hidden_layer_sizes[i]);
        init2DMatrix(Hs[i],minibatch_size,hidden_layer_sizes[i]);
        // Statistic data
        number_of_weights += former_width*hidden_layer_sizes[i];
        number_of_biases += hidden_layer_sizes[i];
        number_of_hvalues += minibatch_size*hidden_layer_sizes[i];
        // The 2 in front of every equation is because there's a gradient descend version of each matrix
        size_of_Ws += 2*(former_width*hidden_layer_sizes[i]*sizeof(float) + size_of_a_matrix);
        size_of_bs += 2*(hidden_layer_sizes[i]*sizeof(float) + size_of_a_matrix);
        size_of_Hs += 2*(minibatch_size*hidden_layer_sizes[i]*sizeof(float) + size_of_a_matrix);

        // Initialize variables for optimization
        if (use_momentum_update) {
            vWs[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrixZero(vWs[i],former_width,hidden_layer_sizes[i]);
            vbs[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrixZero(vbs[i],1,hidden_layer_sizes[i]);
        }
        if (use_nag_update) {
            vWs[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrixZero(vWs[i],former_width,hidden_layer_sizes[i]);
            vbs[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrixZero(vbs[i],1,hidden_layer_sizes[i]);
            vW_prevs[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrixZero(vW_prevs[i],former_width,hidden_layer_sizes[i]);
            vb_prevs[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrixZero(vb_prevs[i],1,hidden_layer_sizes[i]);
        }
        if (use_rmsprop) {
            Wcaches[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrixZero(Wcaches[i],former_width,hidden_layer_sizes[i]);
            bcaches[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrixZero(bcaches[i],1,hidden_layer_sizes[i]);
        }
        former_width = hidden_layer_sizes[i];
    }
    
    printf("INFO: %d W matrixes, %d learnable weights initialized, %.2f KB meomry used\n", network_depth, number_of_weights, size_of_Ws/1024.0f);
    printf("INFO: %d b matrixes, %d learnable biases initialized, %.2f KB meomry used\n", network_depth, number_of_biases, size_of_bs/1024.0f);
    printf("INFO: %d H matrixes, %d learnable hidden layer values initialized, %.2f KB meomry used\n", network_depth, number_of_hvalues, size_of_Hs/1024.0f);
    printf("INFO: A total number of %.2f KB memory is used by learnable parameters in the network\n",(size_of_Ws+size_of_bs+size_of_Hs)/1024.0f);

    // Feed data to the network to train it
    int iterations = training_data->height / minibatch_size;
    TwoDMatrix* X = matrixMalloc(sizeof(TwoDMatrix));
    for(int epoch=0;epoch<epochs;epoch++) {
        // find number of minibatch_size example to go into the network as 1 iteration
        for(int iteration=0;iteration<iterations;iteration++) {
            int data_start = iteration*minibatch_size;
            int data_end = (iteration+1)*minibatch_size-1;
            chop2DMatrix(training_data,data_start,data_end,X);
            // Forward propagation
            TwoDMatrix* layer_X = NULL;
            layer_X = X;
            for(int i=0;i<network_depth;i++) {
                affineLayerForward(layer_X,Ws[i],bs[i],Hs[i]);
                // The last layer in the network will calculate the scores
                // So there will not be a activation function put to it
                if (i != network_depth - 1) {
                    leakyReLUForward(Hs[i],alpha,Hs[i]);
                }
                debugPrintMatrix(layer_X);
                debugPrintMatrix(Ws[i]);
                debugPrintMatrix(bs[i]);
                debugPrintMatrix(Hs[i]);
                layer_X = Hs[i];
            }
            
            
            float data_loss = softmaxLoss(Hs[network_depth-1], correct_labels, dHs[network_depth-1]);
            debugPrintMatrix(dHs[network_depth-1]);
            float reg_loss = L2RegLoss(Ws, network_depth, reg_strength);
            float loss = data_loss + reg_loss;
            if (epoch % 1000 == 0 && iteration == 0) {
                printf("INFO: Epoch %d, data loss: %f, regulization loss: %f, total loss: %f\n",
                    epoch, data_loss, reg_loss, loss);
            }
            // Backward propagation
            // This dX is only a placeholder to babysit the backword function, of course we are not going to modify X
            TwoDMatrix* dX = matrixMalloc(sizeof(TwoDMatrix));
            for (int i=network_depth-1; i>=0; i--) {
                debugPrintMatrix(dHs[i]);
                debugPrintMatrix(Hs[i]);
                if (i != network_depth-1) {
                    leakyReLUBackward(dHs[i],Hs[i],alpha,dHs[i]);
                }
                debugPrintMatrix(dHs[i]);
                if (i != 0) {
                    affineLayerBackword(dHs[i],Hs[i-1],Ws[i],bs[i],dHs[i-1],dWs[i],dbs[i]);
                } else {
                    affineLayerBackword(dHs[i],X,Ws[i],bs[i],dX,dWs[i],dbs[i]);
                }
                debugPrintMatrix(dWs[i]);
                debugPrintMatrix(Ws[i]);
                // Weight changes contributed by L2 regulization
                L2RegLossBackward(dWs[i],Ws[i],reg_strength,dWs[i]);
                debugPrintMatrix(dWs[i]);
            }
            destroy2DMatrix(dX);
            // Update weights
            for (int i=0;i<network_depth;i++) {
                if (use_momentum_update) {
                    momentumUpdate(Ws[i], dWs[i], vWs[i], mu, learning_rate, Ws[i]);
                    momentumUpdate(bs[i], dbs[i], vbs[i], mu, learning_rate, bs[i]);
                } else if (use_nag_update) {
                    NAGUpdate(Ws[i], dWs[i], vWs[i], vW_prevs[i], mu, learning_rate, Ws[i]);
                    NAGUpdate(bs[i], dbs[i], vbs[i], vb_prevs[i], mu, learning_rate, bs[i]);
                } else if (use_rmsprop) {
                    RMSProp(Ws[i], dWs[i], Wcaches[i], learning_rate, decay_rate, eps, Ws[i]);
                    RMSProp(bs[i], dbs[i], bcaches[i], learning_rate, decay_rate, eps, bs[i]);
                } else {
                    vanillaUpdate(Ws[i],dWs[i],learning_rate,Ws[i]);
                    vanillaUpdate(bs[i],dbs[i],learning_rate,bs[i]);
                }
            }
        }
    }
    // Verify the result with training data
    float correctness = verifyWithTrainingData(
        training_data,
        Ws,
        bs,
        network_depth,
        minibatch_size, 
        alpha,
        labels,
        correct_labels);
    printf("INFO: %f%% correct on training data\n",correctness);
    // Shutdown
    destroy2DMatrix(X);
    for(int i=0;i<network_depth;i++) {
        destroy2DMatrix(Ws[i]);
        destroy2DMatrix(dWs[i]);
        destroy2DMatrix(bs[i]);
        destroy2DMatrix(dbs[i]);
        destroy2DMatrix(Hs[i]);
        destroy2DMatrix(dHs[i]);
        if (use_momentum_update) {
            destroy2DMatrix(vWs[i]);
            destroy2DMatrix(vbs[i]);
        }
        if (use_nag_update) {
            destroy2DMatrix(vWs[i]);
            destroy2DMatrix(vbs[i]);
            destroy2DMatrix(vW_prevs[i]);
            destroy2DMatrix(vb_prevs[i]);
        }
        if (use_rmsprop) {
            destroy2DMatrix(Wcaches[i]);
            destroy2DMatrix(bcaches[i]);
        }
    }
    free(Ws);
    free(dWs);
    free(bs);
    free(dbs);
    free(Hs);
    free(dHs);
    if (use_momentum_update) {
        free(vWs);
        free(vbs);
    }
    if (use_nag_update) {
        free(vWs);
        free(vbs);
        free(vW_prevs);
        free(vb_prevs);
    }
    if (use_rmsprop) {
        free(Wcaches);
        free(bcaches);
    }
    // Remeber to free struct parameter
    destroy2DMatrix(network_params->X);
    destroy2DMatrix(network_params->correct_labels);
    free(network_params->hidden_layer_sizes);
    network_params->hidden_layer_sizes = NULL;
    free(network_params);
    network_params = NULL;
    return 0;
}

int test(TwoDMatrix* X, TwoDMatrix** Ws, TwoDMatrix** bs, float alpha, int network_depth, TwoDMatrix* scores) {
    TwoDMatrix** Hs = malloc(sizeof(TwoDMatrix*)*network_depth);
    for(int i=0;i<network_depth;i++) Hs[i] = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* layer_X = NULL;
    layer_X = X;
    for(int i=0;i<network_depth;i++) {
        affineLayerForward(layer_X,Ws[i],bs[i],Hs[i]);
        if (i != network_depth - 1) {
            leakyReLUForward(Hs[i],alpha,Hs[i]);
        }
        layer_X = Hs[i];
    }
    init2DMatrix(scores,Hs[network_depth-1]->height,Hs[network_depth-1]->width);
    copyTwoDMatrix(Hs[network_depth-1],scores);
    for(int i=0;i<network_depth;i++) destroy2DMatrix(Hs[i]);
    free(Hs);
    Hs = NULL;
    return 0;
}

float verifyWithTrainingData(TwoDMatrix* training_data, TwoDMatrix** Ws, TwoDMatrix** bs, int network_depth, int minibatch_size, float alpha, int labels, TwoDMatrix* correct_labels) {
    int correct_count = 0;
    int iterations = training_data->height / minibatch_size;
    TwoDMatrix* X = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* scores = matrixMalloc(sizeof(TwoDMatrix));
    for(int i=0;i<iterations;i++) {
        int data_start = i*minibatch_size;
        int data_end = (i+1)*minibatch_size-1;
        chop2DMatrix(training_data,data_start,data_end,X);
        test(X,Ws,bs, alpha,network_depth,scores);
        for(int j=data_start;j<=data_end;j++) {
            int correct_label = correct_labels->d[j][0];
            int predicted = 0;
            float max_score = -1e99;
            int score_index = j - data_start;
            for(int k=0;k<labels;k++) {
                if (scores->d[score_index][k] > max_score) {
                    predicted = k;
                    max_score = scores->d[score_index][k];
                }
            }
            if (correct_label == predicted) correct_count++;
        }
    }
    destroy2DMatrix(X);
    destroy2DMatrix(scores);
    return 100.0f*correct_count/(iterations*minibatch_size);
}
