#ifndef __FULLY_CONNECTED_HEADER__
#define __FULLY_CONNECTED_HEADER__

int train_multithread(FCParameters* network_params);
int selftest_MT(TwoDMatrix* X, TwoDMatrix** Ws, TwoDMatrix** bs, float alpha, int network_depth, bool use_batchnorm, TwoDMatrix** mean_caches, TwoDMatrix** var_caches, float eps, TwoDMatrix** gammas, TwoDMatrix** betas, TwoDMatrix* scores, int number_of_threads);
float verifyWithTrainingData_MT(TwoDMatrix* training_data, TwoDMatrix** Ws, TwoDMatrix** bs, int network_depth, int minibatch_size, float alpha, int labels, bool use_batchnorm, TwoDMatrix** mean_caches, TwoDMatrix** var_caches, float eps, TwoDMatrix** gammas, TwoDMatrix** betas, TwoDMatrix* correct_labels, int number_of_threads);
int test_multithread(FCParameters* network_params, TwoDMatrix* scores, int number_of_threads);
int FCTrainCore_multithread(FCParameters* network_params, 
    TwoDMatrix** Ws, TwoDMatrix** bs, 
    TwoDMatrix** vWs, TwoDMatrix** vbs, TwoDMatrix** vW_prevs, TwoDMatrix** vb_prevs,
    TwoDMatrix** Wcaches, TwoDMatrix** bcaches,
    TwoDMatrix** mean_caches, TwoDMatrix** var_caches, TwoDMatrix** gammas, TwoDMatrix** betas,
    TwoDMatrix* dX, int e, float *learning_rate, float* losses, int number_of_threads);

#endif
