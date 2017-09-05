#ifndef __CONVNET_OPTS_HEADER__
#define __CONVNET_OPTS_HEADER__

int calcOutputSize(int length, int padding, int filter_length, int stride);
int zeroPadding(ThreeDMatrix* X, int padding_height, int padding_width, ThreeDMatrix* out);
int unpad(ThreeDMatrix* padded, int padding_height, int padding_width, ThreeDMatrix* out);
int convSingleFilter(ThreeDMatrix* X,ThreeDMatrix* F,ThreeDMatrix* b, int stride_y, int stride_x,float** out);
int maxPoolingSingleSlice(ThreeDMatrix* X, int pooling_height, int pooling_width, int stride_y, int stride_x,int z, float** out);
int maxPoolingSingleSliceBackword(ThreeDMatrix* X, ThreeDMatrix* dV, int pooling_height, int pooling_width, int stride_y, int stride_x,int z, float** out);
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
int convReLUBackword(ThreeDMatrix* dV, ThreeDMatrix* X, float alpha, ThreeDMatrix* dX);
int reshapeThreeDMatrix2Col(ThreeDMatrix* X, int index, TwoDMatrix* OUT);
int restoreThreeDMatrixFromCol(TwoDMatrix* IN, ThreeDMatrix** OUT);

#endif
