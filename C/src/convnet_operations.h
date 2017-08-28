#ifndef __CONVNET_OPTS_HEADER__
#define __CONVNET_OPTS_HEADER__

int calcOutputSize(int length, int padding, int filter_length, int stride);
ThreeDMatrix* zeroPadding(ThreeDMatrix* X, int padding_height, int padding_width);
int unpad(ThreeDMatrix* padded, int padding_height, int padding_width, ThreeDMatrix* out);
int convSingleFilter(ThreeDMatrix* X,ThreeDMatrix* F,ThreeDMatrix* b, int stride_y, int stride_x,float** out);
int maxPoolingSingleSlice(ThreeDMatrix* X, int pooling_height, int pooling_width, int stride_y, int stride_x,int z, float** out);
int convSingleFilterBackward(ThreeDMatrix* X,
    ThreeDMatrix* F, 
    ThreeDMatrix* dV, 
    int stride_y, 
    int stride_x, 
    int z, 
    ThreeDMatrix* dX, 
    ThreeDMatrix* dF, 
    ThreeDMatrix* db);

int convReLUForward(ThreeDMatrix* X, float alpha, ThreeDMatrix* V);
int convReLUBackword(ThreeDMatrix* dX, ThreeDMatrix* X, float alpha, ThreeDMatrix* dV);
int threeDMatrix2Col(ThreeDMatrix* X, TwoDMatrix* OUT);

#endif
