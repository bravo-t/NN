#ifndef __FULLY_CONNECTED_MT_HEADER__
#define __FULLY_CONNECTED_MT_HEADER__

#include "thread_barrier.h"
#include "thread_control.h"

typedef struct {
    ThreadControl* handle;
    int thread_id;
    int network_depth;
    TwoDMatrix* X;
    TwoDMatrix** Ws;
    TwoDMatrix** bs;
    TwoDMatrix** Hs;
    TwoDMatrix** dWs;
    TwoDMatrix** dbs;
    TwoDMatrix** dHs;
    TwoDMatrix** Wcaches;
    TwoDMatrix** bcaches;
    TwoDMatrix* correct_labels;
    float alpha;
    float learning_rate;
    float reg_strength;
    float decay_rate;
    float eps;
    bool use_rmsprop;
    bool* mem_allocated;
    int number_of_threads;
    pthread_mutex_t* mutex;
    pthread_cond_t* cond;
    thread_barrier_t* barrier;
    int* int_retval;
    float* float_retval;
} SlaveArgs;

#endif
