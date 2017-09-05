#ifndef __FULLY_CONNECTED_HEADER__
#define __FULLY_CONNECTED_HEADER__

int train(FCParameters* network_params);
int selftest(TwoDMatrix* X, TwoDMatrix** Ws, TwoDMatrix** bs, float alpha, int network_depth, bool use_batchnorm, TwoDMatrix** mean_caches, TwoDMatrix** var_caches, float eps, TwoDMatrix** gammas, TwoDMatrix** betas, TwoDMatrix* scores);
float verifyWithTrainingData(TwoDMatrix* training_data, TwoDMatrix** Ws, TwoDMatrix** bs, int network_depth, int minibatch_size, float alpha, int labels, bool use_batchnorm, TwoDMatrix** mean_caches, TwoDMatrix** var_caches, float eps, TwoDMatrix** gammas, TwoDMatrix** betas, TwoDMatrix* correct_labels);
int test(FCParameters* network_params);
float* FCTrainCore(FCParameters* network_params, 
    TwoDMatrix** Ws, TwoDMatrix** bs, 
    TwoDMatrix* vWs, TwoDMatrix* vbs, TwoDMatrix* vW_prevs, TwoDMatrix* vb_prevs,
    TwoDMatrix* Wcaches, TwoDMatrix* bcaches,
    TwoDMatrix** mean_caches, TwoDMatrix** var_caches, TwoDMatrix** gammas, TwoDMatrix** betas,
    TwoDMatrix* dX, float* losses);

#endif
