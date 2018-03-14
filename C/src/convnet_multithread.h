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
    ThreeDMatrix** CONV_OUT;
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
    float learning_rate;
    bool verbose;
    int id;
    int number_of_threads;
} ConvnetSlaveArgs;
