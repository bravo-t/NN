#ifndef __CONVNET_HEADER__
#define __CONVNET_HEADER__

int trainConvnet(ConvnetParameters* network_params);
int testConvnet(ConvnetParameters* convnet_params, TwoDMatrix* labels);
int testConvnetCore(int M,int N, int number_of_samples,
    int* filter_number,int* filter_stride_x, int* filter_stride_y, int* filter_width, int* filter_height, 
    bool* enable_maxpooling,int* pooling_stride_x,int* pooling_stride_y,int* pooling_width,int* pooling_height,
    int* padding_width, int* padding_height,
    float alpha, bool normalize_data_per_channel, int K,
    ThreeDMatrix**** F,ThreeDMatrix**** b,
    TwoDMatrix** Ws,TwoDMatrix** bs,
    ThreeDMatrix* labels);
    
#endif