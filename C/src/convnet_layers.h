#ifndef __CONVNET_LAYERS_HEADER__
#define __CONVNET_LAYERS_HEADER__

int convLayerForward(ThreeDMatrix* X, ThreeDMatrix** F, int number_of_filters, ThreeDMatrix** b, int f_height, int f_width, int stride_y, int stride_x, int padding_y, int padding_x, float alpha, ThreeDMatrix* V);
int maxPoolingForward(ThreeDMatrix* X, int stride_y, int stride_x, int pooling_width, int pooling_height, ThreeDMatrix* V);
int convLayerBackward(ThreeDMatrix* X, 
    ThreeDMatrix* V,
    ThreeDMatrix** F, 
    ThreeDMatrix* dV, 
    int padding_y, 
    int padding_x, 
    int stride_y, 
    int stride_x, 
    float alpha,
    ThreeDMatrix* dX, 
    ThreeDMatrix** dF, 
    ThreeDMatrix** db);
int maxPoolingBackward(ThreeDMatrix* dV, ThreeDMatrix* X, int stride_y, int stride_x, int pooling_width, int pooling_height, ThreeDMatrix* dX);
int vanillaUpdateConvnet(
    ThreeDMatrix* X, 
    ThreeDMatrix* dX, 
    float learning_rate, 
    ThreeDMatrix* OUT);

#endif
