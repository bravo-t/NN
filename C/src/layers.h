#ifndef __LAYERS_HEADER__
#define __LAYERS_HEADER__

#define ANSI_BOLD "\x31[1m"
#define ANSI_BOLD_RESET "\x31[0m"
#define ANSI_COLOR_RED "\x1b[31m"
#define ANSI_COLOR_GREEN "\x1b[32m"
#define ANSI_COLOR_RESET "\x1b[0m"
int affineLayerForward(TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* b, TwoDMatrix* OUT);
int affineLayerBackword(TwoDMatrix* dOUT, TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* b, TwoDMatrix* dX, TwoDMatrix* dW, TwoDMatrix* db);
int leakyReLUForward(TwoDMatrix* M, float alpha, TwoDMatrix* OUT);
int leakyReLUBackward(TwoDMatrix* dM, TwoDMatrix* M, float alpha, TwoDMatrix* OUT);
int vanillaUpdate(TwoDMatrix* M, TwoDMatrix* dM, float learning_rate, TwoDMatrix* OUT);
float SVMLoss(TwoDMatrix* score, TwoDMatrix* correct_label, TwoDMatrix* dscore);
float softmaxLoss(TwoDMatrix* score, TwoDMatrix* correct_label, TwoDMatrix* dscore);
float L2RegLoss(TwoDMatrix** Ms,int number_of_weights, float reg_strength);
int L2RegLossBackward(TwoDMatrix* dM, TwoDMatrix* M, float reg_strength, TwoDMatrix* OUT);
int momentumUpdate(TwoDMatrix* X, TwoDMatrix* dX, TwoDMatrix* v, float mu, float learning_rate,  TwoDMatrix* OUT);
int NAGUpdate(TwoDMatrix* X, TwoDMatrix* dX, TwoDMatrix* v, TwoDMatrix* v_prev, float mu, float learning_rate,  TwoDMatrix* OUT);
int RMSProp(TwoDMatrix* X, TwoDMatrix* dX, TwoDMatrix* cache, float learning_rate, float decay_rate, float eps, TwoDMatrix* OUT);
int batchnorm_training_forward(TwoDMatrix* M, float momentum, float eps, TwoDMatrix* gamma, TwoDMatrix* beta, TwoDMatrix* OUT, TwoDMatrix* mean_cache, TwoDMatrix* var_cache, TwoDMatrix* mean, TwoDMatrix* var, TwoDMatrix* M_normalized);
int batchnorm_test_forward(TwoDMatrix* M, TwoDMatrix* mean_cache, TwoDMatrix* var_cache, float eps, TwoDMatrix* gamma, TwoDMatrix* beta, TwoDMatrix* OUT);
int batchnorm_backward(TwoDMatrix* dOUT, TwoDMatrix* M, TwoDMatrix* M_normalized, TwoDMatrix* gamma, TwoDMatrix* beta, TwoDMatrix* mean, TwoDMatrix* var, float eps, TwoDMatrix* dM, TwoDMatrix* dgamma, TwoDMatrix* dbeta);
float training_accuracy(TwoDMatrix* scores, TwoDMatrix* correct_labels);

#endif
