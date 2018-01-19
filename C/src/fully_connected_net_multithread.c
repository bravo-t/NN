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

    // Temporary variables
    TwoDMatrix* X = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* dX = matrixMalloc(sizeof(TwoDMatrix));
    // array that holds data loss and reg loss
    float** losses = malloc(sizeof(float*)*number_of_threads);


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
    thread_barrier_init(&forward_prop_inst_ready,number_of_threads);
    thread_barrier_init(&forward_prop_inst_ack,number_of_threads);
    ThreadControl* forward_prop_control_handle = initControlHandle(&forward_prop_control_handle_mutex, &forward_prop_inst_ready, &forward_prop_inst_ack, number_of_threads);

    bool calc_loss_mem_alloc = false;
    thread_barrier_t calc_loss_barrier = THREAD_BARRIER_INITIALIZER;
    thread_barrier_init(&calc_loss_barrier,number_of_threads);
    pthread_mutex_t calc_loss_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t calc_loss_cond = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t calc_loss_control_handle_mutex = PTHREAD_MUTEX_INITIALIZER;
    thread_barrier_t calc_loss_inst_ready = THREAD_BARRIER_INITIALIZER;
    thread_barrier_t calc_loss_inst_ack = THREAD_BARRIER_INITIALIZER;
    thread_barrier_init(&calc_loss_inst_ready,number_of_threads);
    thread_barrier_init(&calc_loss_inst_ack,number_of_threads);
    ThreadControl* calc_loss_control_handle = initControlHandle(&calc_loss_control_handle_mutex, &calc_loss_inst_ready, &calc_loss_inst_ack, number_of_threads);
    
    bool backward_prop_mem_alloc = false;
    thread_barrier_t backward_prop_barrier = THREAD_BARRIER_INITIALIZER;
    thread_barrier_init(&backward_prop_barrier,number_of_threads);
    pthread_mutex_t backward_prop_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t backward_prop_cond = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t backward_prop_control_handle_mutex = PTHREAD_MUTEX_INITIALIZER;
    thread_barrier_t backward_prop_inst_ready = THREAD_BARRIER_INITIALIZER;
    thread_barrier_t backward_prop_inst_ack = THREAD_BARRIER_INITIALIZER;
    thread_barrier_init(&backward_prop_inst_ready,number_of_threads);
    thread_barrier_init(&backward_prop_inst_ack,number_of_threads);
    ThreadControl* backward_prop_control_handle = initControlHandle(&backward_prop_control_handle_mutex, &backward_prop_inst_ready, &backward_prop_inst_ack, number_of_threads);
    
    bool update_weights_mem_alloc = false;
    thread_barrier_t update_weights_barrier = THREAD_BARRIER_INITIALIZER;
    thread_barrier_init(&update_weights_barrier,number_of_threads);
    pthread_mutex_t update_weights_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t update_weights_cond = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t update_weights_control_handle_mutex = PTHREAD_MUTEX_INITIALIZER;
    thread_barrier_t update_weights_inst_ready = THREAD_BARRIER_INITIALIZER;
    thread_barrier_t update_weights_inst_ack = THREAD_BARRIER_INITIALIZER;
    thread_barrier_init(&update_weights_inst_ready,number_of_threads);
    thread_barrier_init(&update_weights_inst_ack,number_of_threads);
    ThreadControl* update_weights_control_handle = initControlHandle(&update_weights_control_handle_mutex, &update_weights_inst_ready, &update_weights_inst_ack, number_of_threads);
    
    printf("INFO: Creating slave threads\n");

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    pthread_t* forward_prop = malloc(sizeof(pthread_t)*number_of_threads);
    pthread_t* calc_loss = malloc(sizeof(pthread_t)*number_of_threads);
    pthread_t* backward_prop = malloc(sizeof(pthread_t)*number_of_threads);
    pthread_t* update_weights = malloc(sizeof(pthread_t)*number_of_threads);

    SlaveArgs** forward_prop_arguments = malloc(sizeof(SlaveArgs*)*number_of_threads);
    SlaveArgs** calc_loss_arguments = malloc(sizeof(SlaveArgs*)*number_of_threads);
    SlaveArgs** backward_prop_arguments = malloc(sizeof(SlaveArgs*)*number_of_threads);
    SlaveArgs** update_weights_arguments = malloc(sizeof(SlaveArgs*)*number_of_threads);

    for(int i=0;i<number_of_threads;i++) {
        losses[i] = malloc(sizeof(float)*2);

        forward_prop_arguments[i] = (SlaveArgs*) malloc(sizeof(SlaveArgs));
        calc_loss_arguments[i] = (SlaveArgs*) malloc(sizeof(SlaveArgs));
        backward_prop_arguments[i] = (SlaveArgs*) malloc(sizeof(SlaveArgs));
        update_weights_arguments[i] = (SlaveArgs*) malloc(sizeof(SlaveArgs));

        assignSlaveArguments(forward_prop_arguments[i], 
            forward_prop_control_handle,
            i,
            network_depth,
            X,
            Ws,
            bs,
            Hs,
            dWs,
            dbs,
            dHs,
            Wcaches,
            bcaches,
            correct_labels,
            NULL,
            alpha,
            learning_rate,
            reg_strength,
            decay_rate,
            eps,
            use_rmsprop,
            &forward_prop_mem_alloc,
            number_of_threads,
            &forward_prop_mutex,
            &forward_prop_cond,
            &forward_prop_barrier,
            NULL,
            NULL);
        assignSlaveArguments(calc_loss_arguments[i], 
            calc_loss_control_handle,
            i,
            network_depth,
            X,
            Ws,
            bs,
            Hs,
            dWs,
            dbs,
            dHs,
            Wcaches,
            bcaches,
            correct_labels,
            NULL,
            alpha,
            learning_rate,
            reg_strength,
            decay_rate,
            eps,
            use_rmsprop,
            &calc_loss_mem_alloc,
            number_of_threads,
            &calc_loss_mutex,
            &calc_loss_cond,
            &calc_loss_barrier,
            NULL,
            losses[i]);
        assignSlaveArguments(backward_prop_arguments[i], 
            backward_prop_control_handle,
            i,
            network_depth,
            X,
            Ws,
            bs,
            Hs,
            dWs,
            dbs,
            dHs,
            Wcaches,
            bcaches,
            correct_labels,
            dX,
            alpha,
            learning_rate,
            reg_strength,
            decay_rate,
            eps,
            use_rmsprop,
            &backward_prop_mem_alloc,
            number_of_threads,
            &backward_prop_mutex,
            &backward_prop_cond,
            &backward_prop_barrier,
            NULL,
            NULL);
        assignSlaveArguments(update_weights_arguments[i], 
            update_weights_control_handle,
            i,
            network_depth,
            X,
            Ws,
            bs,
            Hs,
            dWs,
            dbs,
            dHs,
            Wcaches,
            bcaches,
            correct_labels,
            NULL,
            alpha,
            learning_rate,
            reg_strength,
            decay_rate,
            eps,
            use_rmsprop,
            &update_weights_mem_alloc,
            number_of_threads,
            &update_weights_mutex,
            &update_weights_cond,
            &update_weights_barrier,
            NULL,
            NULL);

        int create_thread_error;
        create_thread_error = pthread_create(&forward_prop[i],&attr,FCNET_forwardPropagation_slave,&forward_prop_arguments[i]);
        if (create_thread_error) {
            printf("Error happened while creating slave threads\n");
            exit(-1);
        }
        create_thread_error = pthread_create(&calc_loss[i],&attr,FCNET_calcLoss_slave,&calc_loss_arguments[i]);
        if (create_thread_error) {
            printf("Error happened while creating slave threads\n");
            exit(-1);
        }
        create_thread_error = pthread_create(&backward_prop[i],&attr,FCNET_backwardPropagation_slave,&backward_prop_arguments[i]);
        if (create_thread_error) {
            printf("Error happened while creating slave threads\n");
            exit(-1);
        }
        create_thread_error = pthread_create(&update_weights[i],&attr,FCNET_updateWeights_slave,&forward_prop_arguments[i]);
        if (create_thread_error) {
            printf("Error happened while creating slave threads\n");
            exit(-1);
        }

        printf("DEBUG: Slave threads created %d iteration\n", i);
    }

    // Feed data to the network to train it
    printf("INFO: Training network\n");
    int iterations = training_data->height / minibatch_size;
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
            threadController_master(forward_prop_control_handle, THREAD_RESUME);
            
            threadController_master(calc_loss_control_handle, THREAD_RESUME);
            float data_loss = losses[0][0];
            float reg_loss = losses[0][1];
            float accu = training_accuracy(Hs[network_depth-1], correct_labels);
            if ((epoch % 1000 == 0 && iteration == 0) || verbose) {
                printf("INFO: Epoch %d, data loss: %f, regulization loss: %f, total loss: %f, training accuracy: %f\n",
                    epoch, data_loss, reg_loss, data_loss+reg_loss, accu);
            }
            // Backward propagation
            threadController_master(backward_prop_control_handle, THREAD_RESUME);
            // Update weights
            threadController_master(update_weights_control_handle, THREAD_RESUME);
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
        if (use_batchnorm) {
            destroy2DMatrix(gammas[i]);
            destroy2DMatrix(betas[i]);
            destroy2DMatrix(dgammas[i]);
            destroy2DMatrix(dbetas[i]);
            destroy2DMatrix(mean_caches[i]);
            destroy2DMatrix(var_caches[i]);
            destroy2DMatrix(means[i]);
            destroy2DMatrix(vars[i]);
            destroy2DMatrix(Hs_normalized[i]);
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
            leakyReLUForward_thread(Hs[i],alpha,Hs[i],thread_id, mem_allocated,number_of_threads,mutex,cond,barrier);
        }
        layer_X = Hs[i];
    }
    return 0;
}

void* FCNET_forwardPropagation_slave(void* args) {
    SlaveArgs* a = (SlaveArgs*) args;
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
    SlaveArgs* a = (SlaveArgs*) args;
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

int FCNET_backwardPropagation(TwoDMatrix** Ws, TwoDMatrix** Hs, TwoDMatrix** bs, TwoDMatrix** dWs, TwoDMatrix** dbs, TwoDMatrix** dHs, TwoDMatrix* X, TwoDMatrix* dX, int network_depth, float alpha, float reg_strength, int thread_id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
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
    return 0;
}

void* FCNET_backwardPropagation_slave(void* args) {
    SlaveArgs* a = (SlaveArgs*) args;
    TwoDMatrix** Ws = a->Ws;
    TwoDMatrix** Hs = a->Hs;
    TwoDMatrix** bs = a->bs;
    TwoDMatrix** dWs = a->dWs;
    TwoDMatrix** dHs = a->dHs;
    TwoDMatrix** dbs = a->dbs;
    TwoDMatrix* dX = a->tmp;
    TwoDMatrix* X = a->X;
    float alpha = a->alpha;
    float reg_strength = a->reg_strength;
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
        FCNET_backwardPropagation(Ws,Hs,bs,dWs,dbs,dHs,X,dX,network_depth,alpha,reg_strength,thread_id,mem_allocated,number_of_threads,mutex,cond,barrier);
    }
}

int FCNET_updateWeights(TwoDMatrix** Ws, TwoDMatrix** dWs, TwoDMatrix** bs, TwoDMatrix** dbs, TwoDMatrix** Wcaches, TwoDMatrix** bcaches, float learning_rate, float decay_rate,
    float eps, bool use_rmsprop, int network_depth, int thread_id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    for (int i=0;i<network_depth;i++) {
        if (use_rmsprop) {
            RMSProp_thread(Ws[i], dWs[i], Wcaches[i], learning_rate, decay_rate, eps, Ws[i],thread_id,mem_allocated,number_of_threads,mutex,cond,barrier);
            RMSProp_thread(bs[i], dbs[i], bcaches[i], learning_rate, decay_rate, eps, bs[i],thread_id,mem_allocated,number_of_threads,mutex,cond,barrier);
        } else {
            vanillaUpdate_thread(Ws[i],dWs[i],learning_rate,Ws[i],thread_id,mem_allocated,number_of_threads,mutex,cond,barrier);
            vanillaUpdate_thread(bs[i],dbs[i],learning_rate,bs[i],thread_id,mem_allocated,number_of_threads,mutex,cond,barrier);
        }
    }
    return 0;
}

void* FCNET_updateWeights_slave(void* args) {
    SlaveArgs* a = (SlaveArgs*) args;
    TwoDMatrix** Ws = a->Ws;
    TwoDMatrix** bs = a->bs;
    TwoDMatrix** dWs = a->dWs;
    TwoDMatrix** dbs = a->dbs;
    TwoDMatrix** Wcaches = a->Wcaches;
    TwoDMatrix** bcaches = a->bcaches;
    bool use_rmsprop = a->use_rmsprop;
    float learning_rate = a->learning_rate;
    float decay_rate = a->decay_rate;
    float eps = a->eps;
    int network_depth = a->network_depth;
    int thread_id = a->thread_id;
    bool* mem_allocated = a->mem_allocated;
    ThreadControl* handle = a->handle;
    int number_of_threads = a->number_of_threads;
    pthread_mutex_t* mutex = a->mutex;
    pthread_cond_t* cond = a->cond;
    thread_barrier_t* barrier = a->barrier;
    while(1) {
        threadController_slave(handle);
        FCNET_updateWeights(Ws,dWs,bs,dbs,Wcaches,bcaches,learning_rate,decay_rate,eps,use_rmsprop,network_depth,thread_id,mem_allocated,number_of_threads,mutex,cond,barrier);
    }
}

void assignSlaveArguments(SlaveArgs* args,
    ThreadControl* handle,
    int thread_id,
    int network_depth,
    TwoDMatrix* X,
    TwoDMatrix** Ws,
    TwoDMatrix** bs,
    TwoDMatrix** Hs,
    TwoDMatrix** dWs,
    TwoDMatrix** dbs,
    TwoDMatrix** dHs,
    TwoDMatrix** Wcaches,
    TwoDMatrix** bcaches,
    TwoDMatrix* correct_labels,
    TwoDMatrix* tmp,
    float alpha,
    float learning_rate,
    float reg_strength,
    float decay_rate,
    float eps,
    bool use_rmsprop,
    bool* mem_allocated,
    int number_of_threads,
    pthread_mutex_t* mutex,
    pthread_cond_t* cond,
    thread_barrier_t* barrier,
    int* int_retval,
    float* float_retval) {
    args->handle = handle;
    args->thread_id = thread_id;
    args->network_depth = network_depth;
    args->X = X;
    args->Ws = Ws;
    args->bs = bs;
    args->Hs = Hs;
    args->dWs = dWs;
    args->dbs = dbs;
    args->dHs = dHs;
    args->Wcaches = Wcaches;
    args->bcaches = bcaches;
    args->correct_labels = correct_labels;
    args->tmp = tmp;
    args->alpha = alpha;
    args->learning_rate = learning_rate;
    args->reg_strength = reg_strength;
    args->decay_rate = decay_rate;
    args->eps = eps;
    args->use_rmsprop = use_rmsprop;
    args->mem_allocated = mem_allocated;
    args->number_of_threads = number_of_threads;
    args->mutex = mutex;
    args->cond = cond;
    args->barrier = barrier;
    args->int_retval = int_retval;
    args->float_retval = float_retval;
}

int test_multithread(FCParameters* network_params,TwoDMatrix* scores, int number_of_threads) {
    return 0;
}