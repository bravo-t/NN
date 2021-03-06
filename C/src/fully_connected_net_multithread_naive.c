#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <stdarg.h>
#include <stdbool.h>
#include <string.h>
#include "network_type.h"
#include "matrix_operations.h"
#include "matrix_operations_multithread.h"
#include "layers.h"
#include "layers_multithread.h"
#include "misc_utils.h"
#include "fully_connected_net.h"
#include "fully_connected_net_multithread.h"

int train_multithread(FCParameters* network_params) {
    TwoDMatrix* training_data = network_params->X;
    TwoDMatrix* correct_labels = network_params->correct_labels;
    int minibatch_size = network_params->minibatch_size;
    int labels = network_params->labels;
    float reg_strength = network_params->reg_strength;
    float alpha = network_params->alpha;
    float learning_rate = network_params->learning_rate;
    float base_learning_rate = learning_rate;
    int network_depth = network_params->network_depth;
    int* hidden_layer_sizes = network_params->hidden_layer_sizes;
    int epochs = network_params->epochs;

    bool enable_learning_rate_step_decay = network_params->enable_learning_rate_step_decay;
    bool enable_learning_rate_exponential_decay = network_params->enable_learning_rate_exponential_decay;
    bool enable_learning_rate_invert_t_decay = network_params->enable_learning_rate_invert_t_decay;
    int learning_rate_decay_unit = network_params->learning_rate_decay_unit;
    float learning_rate_decay_a0 = network_params->learning_rate_decay_a0;
    float learning_rate_decay_k = network_params->learning_rate_decay_k;
    int shuffle_training_samples = network_params->shuffle_training_samples;
    int save_checkpoint = network_params->save_checkpoint;

    bool verbose = network_params->verbose;
    // Below are control variables for optimizers
    bool use_momentum_update =  network_params->use_momentum_update;
    bool use_nag_update =  network_params->use_nag_update;
    bool use_rmsprop =  network_params->use_rmsprop;
    float mu =  network_params->mu; // or 0.5,0.95, 0.99
    float decay_rate =  network_params->decay_rate; // or with more 9s in it
    float eps =  network_params->eps;

    bool use_batchnorm =  network_params->use_batchnorm;
    float batchnorm_momentum =  network_params->batchnorm_momentum;
    float batchnorm_eps =  network_params->batchnorm_eps;

    int number_of_threads = network_params->number_of_threads;
    printf("INFO: This network consists of %d hidden layers, and their sizes are configured to be ", network_depth);
    for(int i=0;i<network_depth;i++) {
        printf("%d ",hidden_layer_sizes[i]);
    }
    printf("\n");
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
    // Batch normalization layers;
    TwoDMatrix** gammas = NULL;
    TwoDMatrix** betas = NULL;
    TwoDMatrix** dgammas = NULL;
    TwoDMatrix** dbetas = NULL;
    TwoDMatrix** mean_caches = NULL;
    TwoDMatrix** var_caches = NULL;
    TwoDMatrix** means = NULL;
    TwoDMatrix** vars = NULL;
    TwoDMatrix** Hs_normalized = NULL;
    TwoDMatrix** vgammas = NULL;
    TwoDMatrix** vgamma_prevs = NULL;
    TwoDMatrix** vbetas = NULL;
    TwoDMatrix** vbeta_prevs = NULL;
    TwoDMatrix** gamma_caches = NULL;
    TwoDMatrix** beta_caches = NULL;

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
    if (use_batchnorm) {
        printf("INFO: Batch normalization is used\n");
        gammas = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
        betas = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
        dgammas = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
        dbetas = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
        means = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
        vars = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
        mean_caches = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
        var_caches = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
        Hs_normalized = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
        if (use_momentum_update) {
            vgammas = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
            vbetas = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
        }
        if (use_nag_update) {
            vgammas = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
            vbetas = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
            vgamma_prevs = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
            vbeta_prevs = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
        }
        if (use_rmsprop) {
            gamma_caches = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
            beta_caches = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
        }
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
        if (verbose) {
            printf("INFO: Initializing W%d to be a %dx%d matrix\n",i,former_width,hidden_layer_sizes[i]);
            printf("INFO: Initializing b%d to be a %dx%d matrix\n",i,1,hidden_layer_sizes[i]);
            printf("INFO: Initializing H%d to be a %dx%d matrix\n",i,minibatch_size,hidden_layer_sizes[i]);
        }
        init2DMatrixNormRand_MT(Ws[i],former_width,hidden_layer_sizes[i],0.0,1.0,former_width,number_of_threads);
        init2DMatrixZero_MT(bs[i],1,hidden_layer_sizes[i], number_of_threads);
        init2DMatrix_MT(Hs[i],minibatch_size,hidden_layer_sizes[i], number_of_threads);
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
            init2DMatrixZero_MT(vWs[i],former_width,hidden_layer_sizes[i], number_of_threads);
            vbs[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrixZero_MT(vbs[i],1,hidden_layer_sizes[i], number_of_threads);
        }
        if (use_nag_update) {
            vWs[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrixZero_MT(vWs[i],former_width,hidden_layer_sizes[i], number_of_threads);
            vbs[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrixZero_MT(vbs[i],1,hidden_layer_sizes[i], number_of_threads);
            vW_prevs[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrixZero_MT(vW_prevs[i],former_width,hidden_layer_sizes[i], number_of_threads);
            vb_prevs[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrixZero_MT(vb_prevs[i],1,hidden_layer_sizes[i], number_of_threads);
        }
        if (use_rmsprop) {
            Wcaches[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrixZero_MT(Wcaches[i],former_width,hidden_layer_sizes[i], number_of_threads);
            bcaches[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrixZero_MT(bcaches[i],1,hidden_layer_sizes[i], number_of_threads);
        }
        if (use_batchnorm) {
            gammas[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrixOne(gammas[i],1,hidden_layer_sizes[i]);
            betas[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrixZero_MT(betas[i],1,hidden_layer_sizes[i], number_of_threads);
            dgammas[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrix_MT(dgammas[i],1,hidden_layer_sizes[i], number_of_threads);
            dbetas[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrix_MT(dbetas[i],1,hidden_layer_sizes[i], number_of_threads);
            means[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrix_MT(means[i],1,hidden_layer_sizes[i], number_of_threads);
            vars[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrix_MT(vars[i],1,hidden_layer_sizes[i], number_of_threads);
            mean_caches[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrixZero_MT(mean_caches[i],1,hidden_layer_sizes[i], number_of_threads);
            var_caches[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrixZero_MT(var_caches[i],1,hidden_layer_sizes[i], number_of_threads);
            Hs_normalized[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrixZero_MT(Hs_normalized[i],minibatch_size,hidden_layer_sizes[i], number_of_threads);
            if (use_momentum_update) {
                vgammas[i] = matrixMalloc(sizeof(TwoDMatrix));
                init2DMatrixZero_MT(vgammas[i],1,hidden_layer_sizes[i], number_of_threads);
                vbetas[i] = matrixMalloc(sizeof(TwoDMatrix));
                init2DMatrixZero_MT(vbetas[i],1,hidden_layer_sizes[i], number_of_threads);
            }
            if (use_nag_update) {
                vgamma_prevs[i] = matrixMalloc(sizeof(TwoDMatrix));
                init2DMatrixZero_MT(vgamma_prevs[i],1,hidden_layer_sizes[i], number_of_threads);
                vbeta_prevs[i] = matrixMalloc(sizeof(TwoDMatrix));
                init2DMatrixZero_MT(vbeta_prevs[i],1,hidden_layer_sizes[i], number_of_threads);
                vgammas[i] = matrixMalloc(sizeof(TwoDMatrix));
                init2DMatrixZero_MT(vgammas[i],1,hidden_layer_sizes[i], number_of_threads);
                vbetas[i] = matrixMalloc(sizeof(TwoDMatrix));
                init2DMatrixZero_MT(vbetas[i],1,hidden_layer_sizes[i], number_of_threads);
            }
            if (use_rmsprop) {
                gamma_caches[i] = matrixMalloc(sizeof(TwoDMatrix));
                init2DMatrixZero_MT(gamma_caches[i],1,hidden_layer_sizes[i], number_of_threads);
                beta_caches[i] = matrixMalloc(sizeof(TwoDMatrix));
                init2DMatrixZero_MT(beta_caches[i],1,hidden_layer_sizes[i], number_of_threads);
            }
        }
        former_width = hidden_layer_sizes[i];
    }
    
    printf("INFO: %d W matrixes, %d learnable weights initialized, %.2f KB meomry used\n", network_depth, number_of_weights, size_of_Ws/1024.0f);
    printf("INFO: %d b matrixes, %d learnable biases initialized, %.2f KB meomry used\n", network_depth, number_of_biases, size_of_bs/1024.0f);
    printf("INFO: %d H matrixes, %d learnable hidden layer values initialized, %.2f KB meomry used\n", network_depth, number_of_hvalues, size_of_Hs/1024.0f);
    printf("INFO: A total number of %.2f KB memory is used by learnable parameters in the network\n",(size_of_Ws+size_of_bs+size_of_Hs)/1024.0f);

    // Feed data to the network to train it
    printf("INFO: Training network\n");
    int iterations = training_data->height / minibatch_size;
    TwoDMatrix* X = matrixMalloc(sizeof(TwoDMatrix));
    for(int epoch=1;epoch<=epochs;epoch++) {
        learning_rate = decayLearningRate(enable_learning_rate_step_decay,
            enable_learning_rate_exponential_decay,
            enable_learning_rate_invert_t_decay,
            learning_rate_decay_unit,
            learning_rate_decay_k,
            learning_rate_decay_a0,
            epoch,
            base_learning_rate,
            learning_rate);
        // find number of minibatch_size example to go into the network as 1 iteration
        for(int iteration=0;iteration<iterations;iteration++) {
            int data_start = iteration*minibatch_size;
            int data_end = (iteration+1)*minibatch_size-1;
            chop2DMatrix(training_data,data_start,data_end,X);
            // Forward propagation
            TwoDMatrix* layer_X = NULL;
            layer_X = X;
            for(int i=0;i<network_depth;i++) {
                affineLayerForward_MT(layer_X,Ws[i],bs[i],Hs[i], number_of_threads);
                // The last layer in the network will calculate the scores
                // So there will not be a activation function put to it
                if (i != network_depth - 1) {
                    if (use_batchnorm) {
                        batchnorm_training_forward_MT(Hs[i], batchnorm_momentum, batchnorm_eps, gammas[i], betas[i], Hs[i], mean_caches[i], var_caches[i], means[i], vars[i], Hs_normalized[i], number_of_threads);
                    }
                    leakyReLUForward_MT(Hs[i],alpha,Hs[i], number_of_threads);
                }
                //debugPrintMatrix(layer_X);
                //debugPrintMatrix(Ws[i]);
                //debugPrintMatrix(bs[i]);
                //debugPrintMatrix(Hs[i]);
                layer_X = Hs[i];
            }
            
            
            float data_loss = softmaxLoss_MT(Hs[network_depth-1], correct_labels, dHs[network_depth-1], number_of_threads);
            //debugPrintMatrix(dHs[network_depth-1]);
            float reg_loss = L2RegLoss_MT(Ws, network_depth, reg_strength, number_of_threads);
            float loss = data_loss + reg_loss;
            float accu = training_accuracy(Hs[network_depth-1], correct_labels);
            if ((epoch % 1000 == 0 && iteration == 0) || verbose) {
                printf("INFO: Epoch %d, data loss: %f, regulization loss: %f, total loss: %f, training accuracy: %f\n",
                    epoch, data_loss, reg_loss, loss, accu);
            }
            // Backward propagation
            // This dX is only a placeholder to babysit the backword function, of course we are not going to modify X
            TwoDMatrix* dX = matrixMalloc(sizeof(TwoDMatrix));
            for (int i=network_depth-1; i>=0; i--) {
                //debugPrintMatrix(dHs[i]);
                //debugPrintMatrix(Hs[i]);
                if (i != network_depth-1) {
                    leakyReLUBackward_MT(dHs[i],Hs[i],alpha,dHs[i], number_of_threads);
                    if (use_batchnorm) {
                        batchnorm_backward_MT(dHs[i], Hs[i], Hs_normalized[i], gammas[i], betas[i], means[i], vars[i], batchnorm_eps, dHs[i],  dgammas[i], dbetas[i], number_of_threads);
                    }
                }
                //debugPrintMatrix(dHs[i]);
                if (i != 0) {
                    affineLayerBackword_MT(dHs[i],Hs[i-1],Ws[i],bs[i],dHs[i-1],dWs[i],dbs[i], number_of_threads);
                } else {
                    affineLayerBackword_MT(dHs[i],X,Ws[i],bs[i],dX,dWs[i],dbs[i], number_of_threads);
                }
                //debugPrintMatrix(dWs[i]);
                //debugPrintMatrix(Ws[i]);
                // Weight changes contributed by L2 regulization
                L2RegLossBackward_MT(dWs[i],Ws[i],reg_strength,dWs[i], number_of_threads);
                //debugPrintMatrix(dWs[i]);
            }
            destroy2DMatrix_MT(dX, number_of_threads);
            // Update weights
            if (0) {
                printf("INFO: Epoch %d, updating weights with learning rate %f\n",
                    epoch, learning_rate);
            }
            for (int i=0;i<network_depth;i++) {
                if (use_momentum_update) {
                    momentumUpdate_MT(Ws[i], dWs[i], vWs[i], mu, learning_rate, Ws[i], number_of_threads);
                    momentumUpdate_MT(bs[i], dbs[i], vbs[i], mu, learning_rate, bs[i], number_of_threads);
                    //if (use_batchnorm) {
                    //    momentumUpdate(gammas[i],dgammas[i],)
                    //}
                } else if (use_nag_update) {
                    NAGUpdate_MT(Ws[i], dWs[i], vWs[i], vW_prevs[i], mu, learning_rate, Ws[i], number_of_threads);
                    NAGUpdate_MT(bs[i], dbs[i], vbs[i], vb_prevs[i], mu, learning_rate, bs[i], number_of_threads);
                } else if (use_rmsprop) {
                    RMSProp_MT(Ws[i], dWs[i], Wcaches[i], learning_rate, decay_rate, eps, Ws[i], number_of_threads);
                    RMSProp_MT(bs[i], dbs[i], bcaches[i], learning_rate, decay_rate, eps, bs[i], number_of_threads);
                } else {
                    vanillaUpdate_MT(Ws[i],dWs[i],learning_rate,Ws[i], number_of_threads);
                    vanillaUpdate_MT(bs[i],dbs[i],learning_rate,bs[i], number_of_threads);
                }
                // Let's just use normal SGD update for batchnorm parameters to make it simpler
                if (use_batchnorm) {
                    vanillaUpdate_MT(gammas[i],dgammas[i],learning_rate,gammas[i], number_of_threads);
                    vanillaUpdate_MT(betas[i],dbetas[i],learning_rate,betas[i], number_of_threads);
                }
            }
        }
        if (shuffle_training_samples != 0 && epoch % shuffle_training_samples == 0) {
            shuffleTrainingSamplesFCNet(training_data, correct_labels, training_data, correct_labels);
        }
        if (save_checkpoint != 0 && epoch != 0 && epoch % save_checkpoint == 0) {
            // Save checkpoints
            char checkpoint_counter[1000];
            sprintf(checkpoint_counter,"%d",epoch/save_checkpoint);
            int checkpoint_length = 20 + strlen(checkpoint_counter);
            char* checkpoint_filename = malloc(sizeof(char)*checkpoint_length);
            strcpy(checkpoint_filename,"checkpoint_");
            strcat(checkpoint_filename,checkpoint_counter);
            strcat(checkpoint_filename,".params");
            dumpNetworkConfig(network_depth, alpha, Ws, bs, use_batchnorm, mean_caches, var_caches, gammas, betas, batchnorm_eps, network_params->params_save_dir,checkpoint_filename);
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
        use_batchnorm,
        mean_caches,
        var_caches,
        eps,
        gammas,
        betas,
        correct_labels);
    printf("INFO: %f%% correct on training data\n",correctness);

    // Dump the whole network configuration for testing
    dumpNetworkConfig(network_depth, alpha, Ws, bs, use_batchnorm, mean_caches, var_caches, gammas, betas, batchnorm_eps, network_params->params_save_dir,"network.params");

    // Shutdown
    destroy2DMatrix_MT(X, number_of_threads);
    for(int i=0;i<network_depth;i++) {
        destroy2DMatrix_MT(Ws[i], number_of_threads);
        destroy2DMatrix_MT(dWs[i], number_of_threads);
        destroy2DMatrix_MT(bs[i], number_of_threads);
        destroy2DMatrix_MT(dbs[i], number_of_threads);
        destroy2DMatrix_MT(Hs[i], number_of_threads);
        destroy2DMatrix_MT(dHs[i], number_of_threads);
        if (use_momentum_update) {
            destroy2DMatrix_MT(vWs[i], number_of_threads);
            destroy2DMatrix_MT(vbs[i], number_of_threads);
        }
        if (use_nag_update) {
            destroy2DMatrix_MT(vWs[i], number_of_threads);
            destroy2DMatrix_MT(vbs[i], number_of_threads);
            destroy2DMatrix_MT(vW_prevs[i], number_of_threads);
            destroy2DMatrix_MT(vb_prevs[i], number_of_threads);
        }
        if (use_rmsprop) {
            destroy2DMatrix_MT(Wcaches[i], number_of_threads);
            destroy2DMatrix_MT(bcaches[i], number_of_threads);
        }
        if (use_batchnorm) {
            destroy2DMatrix_MT(gammas[i], number_of_threads);
            destroy2DMatrix_MT(betas[i], number_of_threads);
            destroy2DMatrix_MT(dgammas[i], number_of_threads);
            destroy2DMatrix_MT(dbetas[i], number_of_threads);
            destroy2DMatrix_MT(mean_caches[i], number_of_threads);
            destroy2DMatrix_MT(var_caches[i], number_of_threads);
            destroy2DMatrix_MT(means[i], number_of_threads);
            destroy2DMatrix_MT(vars[i], number_of_threads);
            destroy2DMatrix_MT(Hs_normalized[i], number_of_threads);
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
    destroy2DMatrix_MT(network_params->X, number_of_threads);
    destroy2DMatrix_MT(network_params->correct_labels, number_of_threads);
    free(network_params->hidden_layer_sizes);
    free(network_params->mode);
    free(network_params->params_save_dir);
    free(network_params->params_filename);
    network_params->hidden_layer_sizes = NULL;
    network_params->params_save_dir = NULL;
    network_params->mode = NULL;
    free(network_params);
    network_params = NULL;
    return 0;
}

int selftest_MT(TwoDMatrix* X, TwoDMatrix** Ws, TwoDMatrix** bs, float alpha, int network_depth, bool use_batchnorm, TwoDMatrix** mean_caches, TwoDMatrix** var_caches, float eps, TwoDMatrix** gammas, TwoDMatrix** betas, TwoDMatrix* scores, int number_of_threads) {
    TwoDMatrix** Hs = malloc(sizeof(TwoDMatrix*)*network_depth);
    for(int i=0;i<network_depth;i++) Hs[i] = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* layer_X = NULL;
    layer_X = X;
    for(int i=0;i<network_depth;i++) {
        affineLayerForward_MT(layer_X,Ws[i],bs[i],Hs[i], number_of_threads);
        if (i != network_depth - 1) {
            if (use_batchnorm) {
                batchnorm_test_forward_MT(Hs[i], mean_caches[i], var_caches[i], eps, gammas[i], betas[i], Hs[i], number_of_threads);
            }
            leakyReLUForward_MT(Hs[i],alpha,Hs[i], number_of_threads);
        }
        layer_X = Hs[i];
    }
    init2DMatrix_MT(scores,Hs[network_depth-1]->height,Hs[network_depth-1]->width, number_of_threads);
    copyTwoDMatrix(Hs[network_depth-1],scores);
    for(int i=0;i<network_depth;i++) destroy2DMatrix_MT(Hs[i], number_of_threads);
    free(Hs);
    Hs = NULL;
    return 0;
}

float verifyWithTrainingData_MT(TwoDMatrix* training_data, TwoDMatrix** Ws, TwoDMatrix** bs, int network_depth, int minibatch_size, float alpha, int labels, bool use_batchnorm, TwoDMatrix** mean_caches, TwoDMatrix** var_caches, float eps, TwoDMatrix** gammas, TwoDMatrix** betas, TwoDMatrix* correct_labels, int number_of_threads) {
    int correct_count = 0;
    int iterations = training_data->height / minibatch_size;
    TwoDMatrix* X = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* scores = matrixMalloc(sizeof(TwoDMatrix));
    for(int i=0;i<iterations;i++) {
        int data_start = i*minibatch_size;
        int data_end = (i+1)*minibatch_size-1;
        chop2DMatrix(training_data,data_start,data_end,X);
        selftest_MT(X,Ws,bs, alpha, network_depth, use_batchnorm, mean_caches, var_caches, eps, gammas, betas, scores, number_of_threads);
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

int test_multithread(FCParameters* network_params, TwoDMatrix* scores, int number_of_threads) {
    TwoDMatrix** Ws = NULL;
    TwoDMatrix** bs = NULL;
    TwoDMatrix* test_data = network_params->X;
    float alpha;
    int network_depth;
    bool use_batchnorm;
    float batchnorm_eps;
    TwoDMatrix** gammas = NULL;
    TwoDMatrix** betas = NULL;
    TwoDMatrix** mean_caches = NULL;
    TwoDMatrix** var_caches = NULL;
    loadNetworkConfig(network_params->params_save_dir,network_params->params_filename, &network_depth, &alpha, &Ws, &bs, &use_batchnorm, &mean_caches, &var_caches, &gammas, &betas, &batchnorm_eps);
    selftest_MT(test_data,Ws,bs, alpha, network_depth, use_batchnorm, mean_caches, var_caches, batchnorm_eps, gammas, betas, scores, number_of_threads);
    //printf("Scores are calculated as:\n");
    //printMatrix(scores);
    return 0;
}

int FCTrainCore_multithread(FCParameters* network_params, 
    TwoDMatrix** Ws, TwoDMatrix** bs, 
    TwoDMatrix** vWs, TwoDMatrix** vbs, TwoDMatrix** vW_prevs, TwoDMatrix** vb_prevs,
    TwoDMatrix** Wcaches, TwoDMatrix** bcaches,
    TwoDMatrix** mean_caches, TwoDMatrix** var_caches, TwoDMatrix** gammas, TwoDMatrix** betas,
    TwoDMatrix* dX, int e, float* learning_rate, float* losses, int number_of_threads) {
    TwoDMatrix* training_data = network_params->X;
    TwoDMatrix* correct_labels = network_params->correct_labels;
    int minibatch_size = network_params->minibatch_size;
    //int labels = network_params->labels;
    float reg_strength = network_params->reg_strength;
    float alpha = network_params->alpha;
    float base_learning_rate = network_params->learning_rate;
    int network_depth = network_params->network_depth;
    int* hidden_layer_sizes = network_params->hidden_layer_sizes;
    int epochs = network_params->epochs;
    
    bool enable_learning_rate_step_decay = network_params->enable_learning_rate_step_decay;
    bool enable_learning_rate_exponential_decay = network_params->enable_learning_rate_exponential_decay;
    bool enable_learning_rate_invert_t_decay = network_params->enable_learning_rate_invert_t_decay;
    int learning_rate_decay_unit = network_params->learning_rate_decay_unit;
    float learning_rate_decay_a0 = network_params->learning_rate_decay_a0;
    float learning_rate_decay_k = network_params->learning_rate_decay_k;

    //bool verbose = network_params->verbose;
    // Below are control variables for optimizers
    bool use_momentum_update =  network_params->use_momentum_update;
    bool use_nag_update =  network_params->use_nag_update;
    bool use_rmsprop =  network_params->use_rmsprop;
    float mu =  network_params->mu; // or 0.5,0.95, 0.99
    float decay_rate =  network_params->decay_rate; // or with more 9s in it
    float eps =  network_params->eps;

    bool use_batchnorm =  network_params->use_batchnorm;
    float batchnorm_momentum =  network_params->batchnorm_momentum;
    float batchnorm_eps =  network_params->batchnorm_eps;
    // Initialize all learnable parameters
    // Hidden layers
    TwoDMatrix** Hs = malloc(sizeof(TwoDMatrix*)*network_depth);
    // Gradient descend values of Weights
    TwoDMatrix** dWs = malloc(sizeof(TwoDMatrix*)*network_depth);
    // Gradient descend values of Biases
    TwoDMatrix** dbs = malloc(sizeof(TwoDMatrix*)*network_depth);
    // Gradient descend values of Hidden layers
    TwoDMatrix** dHs = malloc(sizeof(TwoDMatrix*)*network_depth);
    // Below variables are used in optimization algorithms
    // Batch normalization layers;
    TwoDMatrix** dgammas = NULL;
    TwoDMatrix** dbetas = NULL;
    TwoDMatrix** means = NULL;
    TwoDMatrix** vars = NULL;
    TwoDMatrix** Hs_normalized = NULL;

    if (use_batchnorm) {
        dgammas = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
        dbetas = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
        means = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
        vars = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
        Hs_normalized = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
    }

    //int former_width = training_data->width;
    for(int i=0;i<network_depth;i++) {
        // Initialize layer data holders
        Hs[i] = matrixMalloc(sizeof(TwoDMatrix));
        dWs[i] = matrixMalloc(sizeof(TwoDMatrix));
        dbs[i] = matrixMalloc(sizeof(TwoDMatrix));
        dHs[i] = matrixMalloc(sizeof(TwoDMatrix));
        init2DMatrix_MT(Hs[i],minibatch_size,hidden_layer_sizes[i], number_of_threads);

        // Initialize variables for optimization
        if (use_batchnorm) {
            dgammas[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrix_MT(dgammas[i],1,hidden_layer_sizes[i], number_of_threads);
            dbetas[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrix_MT(dbetas[i],1,hidden_layer_sizes[i], number_of_threads);
            means[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrix_MT(means[i],1,hidden_layer_sizes[i], number_of_threads);
            vars[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrix_MT(vars[i],1,hidden_layer_sizes[i], number_of_threads);
            Hs_normalized[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrixZero_MT(Hs_normalized[i],minibatch_size,hidden_layer_sizes[i], number_of_threads);
        }
        //former_width = hidden_layer_sizes[i];
    }
    
    // Feed data to the network to train it
    //printf("%s: Training network\n",TAG);
    int iterations = training_data->height / minibatch_size;
    TwoDMatrix* X = matrixMalloc(sizeof(TwoDMatrix));
    for(int epoch=1;epoch<=epochs;epoch++) {
        *learning_rate = decayLearningRate(enable_learning_rate_step_decay,
            enable_learning_rate_exponential_decay,
            enable_learning_rate_invert_t_decay,
            learning_rate_decay_unit,
            learning_rate_decay_k,
            learning_rate_decay_a0,
            epoch,
            base_learning_rate,
            *learning_rate);
        // find number of minibatch_size example to go into the network as 1 iteration
        for(int iteration=0;iteration<iterations;iteration++) {
            int data_start = iteration*minibatch_size;
            int data_end = (iteration+1)*minibatch_size-1;
            chop2DMatrix(training_data,data_start,data_end,X);
            // Forward propagation
            TwoDMatrix* layer_X = NULL;
            layer_X = X;
            for(int i=0;i<network_depth;i++) {
                affineLayerForward_MT(layer_X,Ws[i],bs[i],Hs[i], number_of_threads);
                // The last layer in the network will calculate the scores
                // So there will not be a activation function put to it
                if (i != network_depth - 1) {
                    if (use_batchnorm) {
                        batchnorm_training_forward_MT(Hs[i], batchnorm_momentum, batchnorm_eps, gammas[i], betas[i], Hs[i], mean_caches[i], var_caches[i], means[i], vars[i], Hs_normalized[i], number_of_threads);
                    }
                    leakyReLUForward_MT(Hs[i],alpha,Hs[i], number_of_threads);
                }
                //printf("%dth Hidden input, X\n",i);
                //printMatrix(layer_X);
                //printf("Ws[%d]\n",i);
                //printMatrix(Ws[i]);
                //printf("bs[%d]\n",i);
                //printMatrix(bs[i]);
                //printf("Hs[%d]\n",i);
                //printMatrix(Hs[i]);
                layer_X = Hs[i];

#if defined(DEBUG) && DEBUG > 1
                /**********************/
                /******* DEBUG ********/
                debugCheckingForNaNs2DMatrix(Ws[i], "after forward prop, Ws", i);
                debugCheckingForNaNs2DMatrix(bs[i], "after forward prop, bs", i);
                debugCheckingForNaNs2DMatrix(Hs[i], "after forward prop, Hs", i);
                /******* DEBUG ********/
                /**********************/
#endif
            }
            
            
            losses[0] = softmaxLoss_MT(Hs[network_depth-1], correct_labels, dHs[network_depth-1], number_of_threads);
            //losses[0] = SVMLoss(Hs[network_depth-1], correct_labels, dHs[network_depth-1]);
            //printf("dscores\n");
            //printMatrix(dHs[network_depth-1]);
            losses[1] = L2RegLoss_MT(Ws, network_depth, reg_strength, number_of_threads);
            losses[2] = training_accuracy(Hs[network_depth-1], correct_labels);
            //if ((epoch % 1000 == 0 && iteration == 0) || verbose) {
            //    printf("%s: Epoch %d, data loss: %f, regulization loss: %f, total loss: %f\n",TAG,
            //        epoch, data_loss, reg_loss, loss);
            //}
            // Backward propagation
            // This dX is only a placeholder to babysit the backword function, of course we are not going to modify X
            for (int i=network_depth-1; i>=0; i--) {
                //printf("dHs[%d]\n",i);
                //printMatrix(dHs[i]);
                //printf("Hs[%d]\n",i);
                //printMatrix(Hs[i]);
                if (i != network_depth-1) {
                    leakyReLUBackward_MT(dHs[i],Hs[i],alpha,dHs[i], number_of_threads);
                    if (use_batchnorm) {
                        batchnorm_backward_MT(dHs[i], Hs[i], Hs_normalized[i], gammas[i], betas[i], means[i], vars[i], batchnorm_eps, dHs[i],  dgammas[i], dbetas[i], number_of_threads);
                    }
                }
                //debugPrintMatrix(dHs[i]);
                if (i != 0) {
                    affineLayerBackword_MT(dHs[i],Hs[i-1],Ws[i],bs[i],dHs[i-1],dWs[i],dbs[i], number_of_threads);
                } else {
                    affineLayerBackword_MT(dHs[i],X,Ws[i],bs[i],dX,dWs[i],dbs[i], number_of_threads);
                }
                //printf("before reg, dbs[%d]\n",i);
                //printMatrix(dbs[i]);
                //printf("before reg, dWs[%d]\n",i);
                //printMatrix(dWs[i]);
                //printf("Ws[%d]\n",i);
                //printMatrix(Ws[i]);
                // Weight changes contributed by L2 regulization
                L2RegLossBackward_MT(dWs[i],Ws[i],reg_strength,dWs[i], number_of_threads);
                //printf("after reg, dWs[%d]\n",i);
                //printMatrix(dWs[i]);
#if defined(DEBUG) && DEBUG > 1
                /**********************/
                /******* DEBUG ********/
                debugCheckingForNaNs2DMatrix(dWs[i], "after forward prop, dWs", i);
                debugCheckingForNaNs2DMatrix(dbs[i], "after forward prop, dbs", i);
                /******* DEBUG ********/
                /**********************/
#endif
            }
            // Update weights
            for (int i=0;i<network_depth;i++) {
                if (use_momentum_update) {
                    momentumUpdate_MT(Ws[i], dWs[i], vWs[i], mu, *learning_rate, Ws[i], number_of_threads);
                    momentumUpdate_MT(bs[i], dbs[i], vbs[i], mu, *learning_rate, bs[i], number_of_threads);
                    //if (use_batchnorm) {
                    //    momentumUpdate(gammas[i],dgammas[i],)
                    //}
                } else if (use_nag_update) {
                    NAGUpdate_MT(Ws[i], dWs[i], vWs[i], vW_prevs[i], mu, *learning_rate, Ws[i], number_of_threads);
                    NAGUpdate_MT(bs[i], dbs[i], vbs[i], vb_prevs[i], mu, *learning_rate, bs[i], number_of_threads);
                } else if (use_rmsprop) {
                    RMSProp_MT(Ws[i], dWs[i], Wcaches[i], *learning_rate, decay_rate, eps, Ws[i], number_of_threads);
                    RMSProp_MT(bs[i], dbs[i], bcaches[i], *learning_rate, decay_rate, eps, bs[i], number_of_threads);
                } else {
                    vanillaUpdate_MT(Ws[i],dWs[i],*learning_rate,Ws[i], number_of_threads);
                    vanillaUpdate_MT(bs[i],dbs[i],*learning_rate,bs[i], number_of_threads);
                }
                // Let's just use normal SGD update for batchnorm parameters to make it simpler
                if (use_batchnorm) {
                    vanillaUpdate_MT(gammas[i],dgammas[i],*learning_rate,gammas[i], number_of_threads);
                    vanillaUpdate_MT(betas[i],dbetas[i],*learning_rate,betas[i], number_of_threads);
                }
            }
        }
    }
    destroy2DMatrix_MT(X, number_of_threads);
    for(int i=0;i<network_depth;i++) {
        destroy2DMatrix_MT(dWs[i], number_of_threads);
        destroy2DMatrix_MT(dbs[i], number_of_threads);
        destroy2DMatrix_MT(Hs[i], number_of_threads);
        destroy2DMatrix_MT(dHs[i], number_of_threads);
        if (use_batchnorm) {
            destroy2DMatrix_MT(dgammas[i], number_of_threads);
            destroy2DMatrix_MT(dbetas[i], number_of_threads);
            destroy2DMatrix_MT(means[i], number_of_threads);
            destroy2DMatrix_MT(vars[i], number_of_threads);
            destroy2DMatrix_MT(Hs_normalized[i], number_of_threads);
        }
    }
    free(dWs);
    free(dbs);
    free(Hs);
    free(dHs);
    return 0;
}
