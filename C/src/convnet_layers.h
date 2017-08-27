#ifndef __CONVNET_LAYERS_HEADER__
#define __CONVNET_LAYERS_HEADER__

ThreeDMatrix* convLayerForward(ThreeDMatrix* X, ThreeDMatrix** F, int number_of_filters, ThreeDMatrix** b, int f_height, int f_width, int stride_y, int stride_x, int padding_y, int padding_x);
ThreeDMatrix* maxPoolingForward(ThreeDMatrix* X, int stride_y, int stride_x, int pooling_width, int pooling_height);
int convLayerBackward(ThreeDMatrix* X, 
    ThreeDMatrix** F, 
    ThreeDMatrix* dV, 
    int padding_y, 
    int padding_x, 
    int stride_y, 
    int stride_x, 
    ThreeDMatrix* dX, 
    ThreeDMatrix** dF, 
    ThreeDMatrix* db);



#endif