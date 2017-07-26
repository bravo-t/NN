#ifndef __LAYERS_HEADER__
#define __LAYERS_HEADER__

int affineLayerForward(TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* b, TwoDMatrix* OUT);
int affineLayerBackword(TwoDMatrix* dOUT, TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* b, TwoDMatrix* dX, TwoDMatrix* dW, TwoDMatrix* db);
int leakyReLUForward(TwoDMatrix* M, float alpha, TwoDMatrix* OUT);
int leakyReLUBackward(TwoDMatrix* dM, TwoDMatrix* M, float alpha, TwoDMatrix* OUT);
int vanillaUpdate(TwoDMatrix* M, TwoDMatrix* dM, float learning_rate, TwoDMatrix* OUT);
float SVMLoss(TwoDMatrix* score, TwoDMatrix* correct_label, TwoDMatrix* dscore);
float softmaxLoss(TwoDMatrix* score, TwoDMatrix* correct_label, TwoDMatrix* dscore);
float L2RegLoss(TwoDMatrix** Ms,int number_of_weights, float reg_strength);
int L2RegLossBackward(TwoDMatrix* dM, TwoDMatrix* M, float reg_strength, TwoDMatrix* OUT);

#endif
