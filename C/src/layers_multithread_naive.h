#ifndef __LAYERS_MT_HEADER__
#define __LAYERS_MT_HEADER__

int affineLayerForward_MT(TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* b, TwoDMatrix* OUT, int number_of_threads);
int affineLayerBackword_MT(TwoDMatrix* dOUT, TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* b, TwoDMatrix* dX, TwoDMatrix* dW, TwoDMatrix* db, int number_of_threads);
int leakyReLUForward_MT(TwoDMatrix* M, float alpha, TwoDMatrix* OUT, int number_of_threads);
int vanillaUpdate_MT(TwoDMatrix* M, TwoDMatrix* dM, float learning_rate, TwoDMatrix* OUT, int number_of_threads);
void* leakyReLUBackwardRow(void* args);
int leakyReLUBackward_MT(TwoDMatrix* dM, TwoDMatrix* M, float alpha, TwoDMatrix* OUT, int number_of_threads);
float softmaxLoss_MT(TwoDMatrix* score, TwoDMatrix* correct_label, TwoDMatrix* dscore, int number_of_threads);
float L2RegLoss_MT(TwoDMatrix** Ms,int network_depth, float reg_strength, int number_of_threads);
int L2RegLossBackward_MT(TwoDMatrix* dM, TwoDMatrix* M, float reg_strength, TwoDMatrix* OUT, int number_of_threads);
int momentumUpdate_MT(TwoDMatrix* X, TwoDMatrix* dX, TwoDMatrix* v, float mu, float learning_rate,  TwoDMatrix* OUT, int number_of_threads);
int NAGUpdate_MT(TwoDMatrix* X, TwoDMatrix* dX, TwoDMatrix* v, TwoDMatrix* v_prev, float mu, float learning_rate,  TwoDMatrix* OUT, int number_of_threads);
int RMSProp_MT(TwoDMatrix* X, TwoDMatrix* dX, TwoDMatrix* cache, float learning_rate, float decay_rate, float eps, TwoDMatrix* OUT, int number_of_threads);
int batchnorm_training_forward_MT(TwoDMatrix* M, float momentum, float eps, TwoDMatrix* gamma, TwoDMatrix* beta, TwoDMatrix* OUT, TwoDMatrix* mean_cache, TwoDMatrix* var_cache, TwoDMatrix* mean, TwoDMatrix* var, TwoDMatrix* M_normalized, int number_of_threads);
int batchnorm_test_forward_MT(TwoDMatrix* M, TwoDMatrix* mean_cache, TwoDMatrix* var_cache, float eps, TwoDMatrix* gamma, TwoDMatrix* beta, TwoDMatrix* OUT, int number_of_threads);
int batchnorm_backward_MT(TwoDMatrix* dOUT, TwoDMatrix* M, TwoDMatrix* M_normalized, TwoDMatrix* gamma, TwoDMatrix* beta, TwoDMatrix* mean, TwoDMatrix* var, float eps, TwoDMatrix* dM, TwoDMatrix* dgamma, TwoDMatrix* dbeta, int number_of_threads);


#endif
