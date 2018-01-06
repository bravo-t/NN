#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <stdarg.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>
#include <semaphore.h>
#include "thread_barrier.h"
#include "thread_control.h"
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
        init2DMatrixNormRand(Ws[i],former_width,hidden_layer_sizes[i],0.0,1.0,former_width);
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
        if (use_batchnorm) {
            gammas[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrixOne(gammas[i],1,hidden_layer_sizes[i]);
            betas[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrixZero(betas[i],1,hidden_layer_sizes[i]);
            dgammas[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrix(dgammas[i],1,hidden_layer_sizes[i]);
            dbetas[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrix(dbetas[i],1,hidden_layer_sizes[i]);
            means[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrix(means[i],1,hidden_layer_sizes[i]);
            vars[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrix(vars[i],1,hidden_layer_sizes[i]);
            mean_caches[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrixZero(mean_caches[i],1,hidden_layer_sizes[i]);
            var_caches[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrixZero(var_caches[i],1,hidden_layer_sizes[i]);
            Hs_normalized[i] = matrixMalloc(sizeof(TwoDMatrix));
            init2DMatrixZero(Hs_normalized[i],minibatch_size,hidden_layer_sizes[i]);
            if (use_momentum_update) {
                vgammas[i] = matrixMalloc(sizeof(TwoDMatrix));
                init2DMatrixZero(vgammas[i],1,hidden_layer_sizes[i]);
                vbetas[i] = matrixMalloc(sizeof(TwoDMatrix));
                init2DMatrixZero(vbetas[i],1,hidden_layer_sizes[i]);
            }
            if (use_nag_update) {
                vgamma_prevs[i] = matrixMalloc(sizeof(TwoDMatrix));
                init2DMatrixZero(vgamma_prevs[i],1,hidden_layer_sizes[i]);
                vbeta_prevs[i] = matrixMalloc(sizeof(TwoDMatrix));
                init2DMatrixZero(vbeta_prevs[i],1,hidden_layer_sizes[i]);
                vgammas[i] = matrixMalloc(sizeof(TwoDMatrix));
                init2DMatrixZero(vgammas[i],1,hidden_layer_sizes[i]);
                vbetas[i] = matrixMalloc(sizeof(TwoDMatrix));
                init2DMatrixZero(vbetas[i],1,hidden_layer_sizes[i]);
            }
            if (use_rmsprop) {
                gamma_caches[i] = matrixMalloc(sizeof(TwoDMatrix));
                init2DMatrixZero(gamma_caches[i],1,hidden_layer_sizes[i]);
                beta_caches[i] = matrixMalloc(sizeof(TwoDMatrix));
                init2DMatrixZero(beta_caches[i],1,hidden_layer_sizes[i]);
            }
        }
        former_width = hidden_layer_sizes[i];
    }
    
    printf("INFO: %d W matrixes, %d learnable weights initialized, %.2f KB meomry used\n", network_depth, number_of_weights, size_of_Ws/1024.0f);
    printf("INFO: %d b matrixes, %d learnable biases initialized, %.2f KB meomry used\n", network_depth, number_of_biases, size_of_bs/1024.0f);
    printf("INFO: %d H matrixes, %d learnable hidden layer values initialized, %.2f KB meomry used\n", network_depth, number_of_hvalues, size_of_Hs/1024.0f);
    printf("INFO: A total number of %.2f KB memory is used by learnable parameters in the network\n",(size_of_Ws+size_of_bs+size_of_Hs)/1024.0f);
    // Create slave workers
    bool forward_prop_mem_alloc = false;
    thread_barrier_t forward_prop_barrier = THREAD_BARRIER_INITIALIZER;
    thread_barrier_init(&forward_prop_barrier,number_of_threads);
    pthread_mutex_t forward_prop_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t forward_prop_cond = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t forward_prop_control_handle_mutex = PTHREAD_MUTEX_INITIALIZER;
    thread_barrier_t forward_prop_inst_ready = THREAD_BARRIER_INITIALIZER;
    thread_barrier_t forward_prop_inst_ack = THREAD_BARRIER_INITIALIZER;
    ThreadControl* forward_prop_control_handle = initControlHandle(&forward_prop_control_handle_mutex, &forward_prop_inst_ready, &forward_prop_inst_ack, number_of_threads);

    bool calc_loss_mem_alloc = false;
    thread_barrier_t calc_loss_barrier = THREAD_BARRIER_INITIALIZER;
    thread_barrier_init(&calc_loss_barrier,number_of_threads);
    pthread_mutex_t calc_loss_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t calc_loss_cond = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t calc_loss_control_handle_mutex = PTHREAD_MUTEX_INITIALIZER;
    thread_barrier_t calc_loss_inst_ready = THREAD_BARRIER_INITIALIZER;
    thread_barrier_t calc_loss_inst_ack = THREAD_BARRIER_INITIALIZER;
    ThreadControl* calc_loss_control_handle = initControlHandle(&calc_loss_control_handle_mutex, &calc_loss_inst_ready, &calc_loss_inst_ack, number_of_threads);
    
    bool backward_prop_mem_alloc = false;
    thread_barrier_t backward_prop_barrier = THREAD_BARRIER_INITIALIZER;
    thread_barrier_init(&backward_prop_barrier,number_of_threads);
    pthread_mutex_t backward_prop_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t backward_prop_cond = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t backward_prop_control_handle_mutex = PTHREAD_MUTEX_INITIALIZER;
    thread_barrier_t backward_prop_inst_ready = THREAD_BARRIER_INITIALIZER;
    thread_barrier_t backward_prop_inst_ack = THREAD_BARRIER_INITIALIZER;
    ThreadControl* backward_prop_control_handle = initControlHandle(&backward_prop_control_handle_mutex, &backward_prop_inst_ready, &backward_prop_inst_ack, number_of_threads);
    
    bool update_weights_mem_alloc = false;
    thread_barrier_t update_weights_barrier = THREAD_BARRIER_INITIALIZER;
    thread_barrier_init(&update_weights_barrier,number_of_threads);
    pthread_mutex_t update_weights_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t update_weights_cond = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t update_weights_control_handle_mutex = PTHREAD_MUTEX_INITIALIZER;
    thread_barrier_t update_weights_inst_ready = THREAD_BARRIER_INITIALIZER;
    thread_barrier_t update_weights_inst_ack = THREAD_BARRIER_INITIALIZER;
    ThreadControl* update_weights_control_handle = initControlHandle(&update_weights_control_handle_mutex, &update_weights_inst_ready, &update_weights_inst_ack, number_of_threads);
    
    printf("INFO: Creating slave threads\n");
    pthread_t* forward_prop = malloc(sizeof(pthread_t)*number_of_threads);
    pthread_t* calc_loss = malloc(sizeof(pthread_t)*number_of_threads);
    pthread_t* backward_prop = malloc(sizeof(pthread_t)*number_of_threads);
    pthread_t* update_weights = malloc(sizeof(pthread_t)*number_of_threads);

    SlaveArgs** forward_prop_arguments = malloc(sizeof(SlaveArgs*)*number_of_threads);
    SlaveArgs** calc_loss_arguments = malloc(sizeof(SlaveArgs*)*number_of_threads);
    SlaveArgs** backward_prop_arguments = malloc(sizeof(SlaveArgs*)*number_of_threads);
    SlaveArgs** update_weights_arguments = malloc(sizeof(SlaveArgs*)*number_of_threads);

    for(int i=0;i<number_of_threads;i++) {
        forward_prop_arguments[i] = (SlaveArgs*) malloc(sizeof(SlaveArgs));
        calc_loss_arguments[i] = (SlaveArgs*) malloc(sizeof(SlaveArgs));
        backward_prop_arguments[i] = (SlaveArgs*) malloc(sizeof(SlaveArgs));
        update_weights_arguments[i] = (SlaveArgs*) malloc(sizeof(SlaveArgs));


    }

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
                affineLayerForward(layer_X,Ws[i],bs[i],Hs[i], number_of_threads);
                // The last layer in the network will calculate the scores
                // So there will not be a activation function put to it
                if (i != network_depth - 1) {
                    if (use_batchnorm) {
                        batchnorm_training_forward(Hs[i], batchnorm_momentum, batchnorm_eps, gammas[i], betas[i], Hs[i], mean_caches[i], var_caches[i], means[i], vars[i], Hs_normalized[i], number_of_threads);
                    }
                    leakyReLUForward(Hs[i],alpha,Hs[i], number_of_threads);
                }
                //debugPrintMatrix(layer_X);
                //debugPrintMatrix(Ws[i]);
                //debugPrintMatrix(bs[i]);
                //debugPrintMatrix(Hs[i]);
                layer_X = Hs[i];
            }
            
            
            float data_loss = softmaxLoss(Hs[network_depth-1], correct_labels, dHs[network_depth-1], number_of_threads);
            //debugPrintMatrix(dHs[network_depth-1]);
            float reg_loss = L2RegLoss(Ws, network_depth, reg_strength, number_of_threads);
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
                    leakyReLUBackward(dHs[i],Hs[i],alpha,dHs[i], number_of_threads);
                    if (use_batchnorm) {
                        batchnorm_backward(dHs[i], Hs[i], Hs_normalized[i], gammas[i], betas[i], means[i], vars[i], batchnorm_eps, dHs[i],  dgammas[i], dbetas[i], number_of_threads);
                    }
                }
                //debugPrintMatrix(dHs[i]);
                if (i != 0) {
                    affineLayerBackword(dHs[i],Hs[i-1],Ws[i],bs[i],dHs[i-1],dWs[i],dbs[i], number_of_threads);
                } else {
                    affineLayerBackword(dHs[i],X,Ws[i],bs[i],dX,dWs[i],dbs[i], number_of_threads);
                }
                //debugPrintMatrix(dWs[i]);
                //debugPrintMatrix(Ws[i]);
                // Weight changes contributed by L2 regulization
                L2RegLossBackward(dWs[i],Ws[i],reg_strength,dWs[i], number_of_threads);
                //debugPrintMatrix(dWs[i]);
            }
            destroy2DMatrix(dX, number_of_threads);
            // Update weights
            if (0) {
                printf("INFO: Epoch %d, updating weights with learning rate %f\n",
                    epoch, learning_rate);
            }
            for (int i=0;i<network_depth;i++) {
                if (use_momentum_update) {
                    momentumUpdate(Ws[i], dWs[i], vWs[i], mu, learning_rate, Ws[i], number_of_threads);
                    momentumUpdate(bs[i], dbs[i], vbs[i], mu, learning_rate, bs[i], number_of_threads);
                    //if (use_batchnorm) {
                    //    momentumUpdate(gammas[i],dgammas[i],)
                    //}
                } else if (use_nag_update) {
                    NAGUpdate(Ws[i], dWs[i], vWs[i], vW_prevs[i], mu, learning_rate, Ws[i], number_of_threads);
                    NAGUpdate(bs[i], dbs[i], vbs[i], vb_prevs[i], mu, learning_rate, bs[i], number_of_threads);
                } else if (use_rmsprop) {
                    RMSProp(Ws[i], dWs[i], Wcaches[i], learning_rate, decay_rate, eps, Ws[i], number_of_threads);
                    RMSProp(bs[i], dbs[i], bcaches[i], learning_rate, decay_rate, eps, bs[i], number_of_threads);
                } else {
                    vanillaUpdate(Ws[i],dWs[i],learning_rate,Ws[i], number_of_threads);
                    vanillaUpdate(bs[i],dbs[i],learning_rate,bs[i], number_of_threads);
                }
                // Let's just use normal SGD update for batchnorm parameters to make it simpler
                if (use_batchnorm) {
                    vanillaUpdate(gammas[i],dgammas[i],learning_rate,gammas[i], number_of_threads);
                    vanillaUpdate(betas[i],dbetas[i],learning_rate,betas[i], number_of_threads);
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
    destroy2DMatrix(X, number_of_threads);
    for(int i=0;i<network_depth;i++) {
        destroy2DMatrix(Ws[i], number_of_threads);
        destroy2DMatrix(dWs[i], number_of_threads);
        destroy2DMatrix(bs[i], number_of_threads);
        destroy2DMatrix(dbs[i], number_of_threads);
        destroy2DMatrix(Hs[i], number_of_threads);
        destroy2DMatrix(dHs[i], number_of_threads);
        if (use_momentum_update) {
            destroy2DMatrix(vWs[i], number_of_threads);
            destroy2DMatrix(vbs[i], number_of_threads);
        }
        if (use_nag_update) {
            destroy2DMatrix(vWs[i], number_of_threads);
            destroy2DMatrix(vbs[i], number_of_threads);
            destroy2DMatrix(vW_prevs[i], number_of_threads);
            destroy2DMatrix(vb_prevs[i], number_of_threads);
        }
        if (use_rmsprop) {
            destroy2DMatrix(Wcaches[i], number_of_threads);
            destroy2DMatrix(bcaches[i], number_of_threads);
        }
        if (use_batchnorm) {
            destroy2DMatrix(gammas[i], number_of_threads);
            destroy2DMatrix(betas[i], number_of_threads);
            destroy2DMatrix(dgammas[i], number_of_threads);
            destroy2DMatrix(dbetas[i], number_of_threads);
            destroy2DMatrix(mean_caches[i], number_of_threads);
            destroy2DMatrix(var_caches[i], number_of_threads);
            destroy2DMatrix(means[i], number_of_threads);
            destroy2DMatrix(vars[i], number_of_threads);
            destroy2DMatrix(Hs_normalized[i], number_of_threads);
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
    destroy2DMatrix(network_params->X, number_of_threads);
    destroy2DMatrix(network_params->correct_labels, number_of_threads);
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

int FCNET_forwardPropagation(TwoDMatrix* X, TwoDMatrix** Ws, TwoDMatrix** bs, TwoDMatrix** Hs, int network_depth, float alpha, int thread_id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    TwoDMatrix* layer_X = NULL;
    layer_X = X;
    for(int i=0;i<network_depth;i++) {
        affineLayerForward_thread(layer_X, Ws[i], bs[i], Hs[i], thread_id, mem_allocated,number_of_threads,mutex,cond,barrier);
        // The last layer in the network will calculate the scores
        // So there will not be a activation function put to it
        if (i != network_depth - 1) {
            leakyReLUForward(Hs[i],alpha,Hs[i],thread_id, mem_allocated,number_of_threads,mutex,cond,barrier);
        }
        layer_X = Hs[i];
    }
}

void* FCNET_forwardPropagation_slave(void* args) {
    (SlaveArgs*) a = (SlaveArgs*) args;
    TwoDMatrix* X = a->X;
    TwoDMatrix** Ws = a->Ws;
    TwoDMatrix** bs = a->bs;
    TwoDMatrix** Hs = a->Hs;
    float alpha = a->alpha;
    int thread_id = a->thread_id;
    bool* mem_allocated = a->mem_allocated;
    int network_depth = a->network_depth;
    ThreadControl* handle = a->handle;
    int number_of_threads = a->number_of_threads;
    pthread_mutex_t* mutex = a->mutex;
    pthread_cond_t* cond = a->cond;
    thread_barrier_t* barrier = a->barrier;
    while(1) {
        threadController_slave(handle);
        FCNET_forwardPropagation(X,Ws,bs,Hs,network_depth,alpha,thread_id,mem_allocated,number_of_threads,mutex,cond,barrier);
    }
}

int FCNET_calcLoss(TwoDMatrix** Ws, TwoDMatrix** Hs, TwoDMatrix* correct_labels, int network_depth, float reg_strength, TwoDMatrix** dHs, float* losses, int thread_id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    losses[0] = softmaxLoss_thread(Hs[network_depth-1], correct_labels, dHs[network_depth-1],thread_id,mem_allocated,number_of_threads,mutex,cond,barrier);
    losses[1] = L2RegLoss_thread(Ws, network_depth, reg_strength,thread_id,mem_allocated,number_of_threads,mutex,cond,barrier);
    return 0;
}

void* FCNET_calcLoss_slave(void* args) {
    (SlaveArgs*) a = (SlaveArgs*) args;
    TwoDMatrix** Ws = a->Ws;
    TwoDMatrix** Hs = a->Hs;
    TwoDMatrix** dHs = a->dHs;
    TwoDMatrix* correct_labels = a->correct_labels;
    float reg_strength = a->reg_strength;
    int thread_id = a->thread_id;
    bool* mem_allocated = a->mem_allocated;
    int network_depth = a->network_depth;
    ThreadControl* handle = a->handle;
    int number_of_threads = a->number_of_threads;
    pthread_mutex_t* mutex = a->mutex;
    pthread_cond_t* cond = a->cond;
    thread_barrier_t* barrier = a->barrier;
    float* losses = a->float_retval;
    while(1) {
        threadController_slave(handle);
        FCNET_calcLoss(Ws,Hs,correct_labels,network_depth,reg_strength,dHs,losses,thread_id,mem_allocated,number_of_threads,mutex,cond,barrier);
    }
}

int FCNET_backwardPropagation(TwoDMatrix** Ws, TwoDMatrix** Hs, TwoDMatrix** bs, TwoDMatrix** dWs, TwoDMatrix** dbs, TwoDMatrix** dHs, int network_depth, float alpha, int thread_id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    for (int i=network_depth-1; i>=0; i--) {
        if (i != network_depth-1) {
            leakyReLUBackward_thread(dHs[i],Hs[i],alpha,dHs[i],thread_id,mem_allocated,number_of_threads,mutex,cond,barrier);
        }
        //debugPrintMatrix(dHs[i]);
        if (i != 0) {
            affineLayerBackword_thread(dHs[i],Hs[i-1],Ws[i],bs[i],dHs[i-1],dWs[i],dbs[i],thread_id,mem_allocated,number_of_threads,mutex,cond,barrier);
        } else {
            affineLayerBackword_thread(dHs[i],X,Ws[i],bs[i],dX,dWs[i],dbs[i],thread_id,mem_allocated,number_of_threads,mutex,cond,barrier);
        }
        //debugPrintMatrix(dWs[i]);
        //debugPrintMatrix(Ws[i]);
        // Weight changes contributed by L2 regulization
        L2RegLossBackward_thread(dWs[i],Ws[i],reg_strength,dWs[i],thread_id,mem_allocated,number_of_threads,mutex,cond,barrier);
        //debugPrintMatrix(dWs[i]);
    }
}

void* FCNET_backwardPropagation_slave(void* args) {
    (SlaveArgs*) a = (SlaveArgs*) args;
    TwoDMatrix** Ws = a->Ws;
    TwoDMatrix** Hs = a->Hs;
    TwoDMatrix** bs = a->bs;
    TwoDMatrix** dWs = a->dWs;
    TwoDMatrix** dHs = a->dHs;
    TwoDMatrix** dbs = a->dbs;
    float alpha = a->alpha;
    int network_depth = a->network_depth;
    int thread_id = a->thread_id;
    bool* mem_allocated = a->mem_allocated;
    int network_depth = a->network_depth;
    ThreadControl* handle = a->handle;
    int number_of_threads = a->number_of_threads;
    pthread_mutex_t* mutex = a->mutex;
    pthread_cond_t* cond = a->cond;
    thread_barrier_t* barrier = a->barrier;
    while(1) {
        threadController_slave(handle);
        FCNET_backwardPropagation(Ws,Hs,bs,dWs,dbs,dHs,network_depth,alpha,thread_id,mem_allocated,number_of_threads,mutex,cond,barrier);
    }
}