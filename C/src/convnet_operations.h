#ifndef __CONVNET_OPTS_HEADER__
#define __CONVNET_OPTS_HEADER__

int calcOutputSize(int length, int padding, int filter_length, int stride);
ThreeDMatrix* zeroPadding(ThreeDMatrix* X, int padding_height, int padding_width);
int convSingleFilter(ThreeDMatrix* X,ThreeDMatrix* F,ThreeDMatrix* b, int stride_y, int stride_x,float** out);
int maxPoolingSingleSlice(ThreeDMatrix* X, int pooling_height, int pooling_width, int stride_y, int stride_x,int z, float** out);


#endif