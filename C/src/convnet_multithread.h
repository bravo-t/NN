#ifndef __CONVNET_MT_HEADER__
#define __CONVNET_MT_HEADER__

#include "thread_barrier.h"
#include "thread_control.h"

typedef struct {
    ThreadControl* handle;
    int thread_id;
    int M;
    int N;
    int minibatch_size;
    ThreeDMatrix** training_data;
    ThreeDMatrix** dX,
    ThreeDMatrix*** CONV_OUT;
    ThreeDMatrix**** C;
    ThreeDMatrix*** P;
    ThreeDMatrix**** F;
    ThreeDMatrix**** b;
    ThreeDMatrix**** dC;
    ThreeDMatrix*** dP;
    ThreeDMatrix**** dF;
    ThreeDMatrix**** db;
    ThreeDMatrix**** Fcache;
    ThreeDMatrix**** bcache;
    int* filter_number;
    int* filter_height;
    int* filter_width;
    int* filter_stride_y;
    int* filter_stride_x;
    int* padding_width;
    int* padding_height;
    bool* enable_maxpooling;
    int* pooling_height;
    int* pooling_width;
    int* pooling_stride_x;
    int* pooling_stride_y;
    float alpha;
    bool use_rmsprop;
    float rmsprop_decay_rate;
    float rmsprop_eps;
    float learning_rate;
    bool verbose;
    int id;
    int number_of_threads;
} ConvnetSlaveArgs;

int trainConvnet_multithread(ConvnetParameters* network_params);
int testConvnet(ConvnetParameters* convnet_params, TwoDMatrix* scores);
int testConvnetCore(ThreeDMatrix** test_data, int M,int N, int number_of_samples,
    int* filter_number,int* filter_stride_x, int* filter_stride_y, int* filter_width, int* filter_height, 
    bool* enable_maxpooling,int* pooling_stride_x,int* pooling_stride_y,int* pooling_width,int* pooling_height,
    int* padding_width, int* padding_height,
    float alpha, bool normalize_data_per_channel, int K,
    ThreeDMatrix**** F,ThreeDMatrix**** b,
    TwoDMatrix** Ws,TwoDMatrix** bs,
    TwoDMatrix* scores);

int CONV_forwardPropagation(int M, int N, int minibatch_size,ThreeDMatrix*** CONV_OUT, ThreeDMatrix**** C, ThreeDMatrix*** P, ThreeDMatrix**** F, ThreeDMatrix**** b, int* filter_number, int* filter_height, int* filter_width, int* filter_stride_y, int* filter_stride_x, int* padding_width, int* padding_height, bool* enable_maxpooling, int* pooling_height, int* pooling_width, int* pooling_stride_x, int* pooling_stride_y, float alpha, bool verbose, int id, int number_of_threads);
int CONV_backwardPropagation(int M, int N, int minibatch_size, ThreeDMatrix**** C, ThreeDMatrix*** P, ThreeDMatrix**** F, ThreeDMatrix**** b, ThreeDMatrix**** dC, ThreeDMatrix*** dP, ThreeDMatrix**** dF, ThreeDMatrix**** db, int* filter_number, int* filter_height, int* filter_width, int* filter_stride_y, int* filter_stride_x, int* padding_width, int* padding_height, bool* enable_maxpooling, int* pooling_height, int* pooling_width, int* pooling_stride_x, int* pooling_stride_y, float alpha, bool verbose, int id,int number_of_threads);
int CONVNET_updateWeights(int M, int N, int minibatch_size, ThreeDMatrix**** C, ThreeDMatrix*** P, ThreeDMatrix**** F, ThreeDMatrix**** b, ThreeDMatrix**** dC, ThreeDMatrix*** dP, ThreeDMatrix**** dF, ThreeDMatrix**** db, ThreeDMatrix**** Fcache, ThreeDMatrix**** bcache, int* filter_number, bool use_rmsprop, int id, int number_of_threads);

void* CONV_forwardPropagation_slave(void* args);
void* CONV_backwardPropagation_slave(void* args);
void* CONV_updateWeights_slave(void* args);

void assignConvSlaveArguments (ConvnetSlaveArgs* args,
    int M, int N, int minibatch_size, ThreeDMatrix*** CONV_OUT,
    ThreeDMatrix**** C, ThreeDMatrix*** P, ThreeDMatrix**** F, ThreeDMatrix**** b, ThreeDMatrix**** dC, ThreeDMatrix*** dP, ThreeDMatrix**** dF, ThreeDMatrix**** db, 
    int* filter_number, int* filter_height, int* filter_width, int* filter_stride_y, int* filter_stride_x, 
    int* padding_width, int* padding_height, 
    bool* enable_maxpooling, int* pooling_height, int* pooling_width, int* pooling_stride_x, int* pooling_stride_y, 
    float alpha, bool verbose, int id,int number_of_threads, ThreadControl* handle);

#endif
