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
    TwoDMatrix* tmp;
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

int FCNET_forwardPropagation(TwoDMatrix* X, TwoDMatrix** Ws, TwoDMatrix** bs, TwoDMatrix** Hs, int network_depth, float alpha, int thread_id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
void* FCNET_forwardPropagation_slave(void* args);
int FCNET_calcLoss(TwoDMatrix** Ws, TwoDMatrix** Hs, TwoDMatrix* correct_labels, int network_depth, float reg_strength, TwoDMatrix** dHs, float* losses, int thread_id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
void* FCNET_calcLoss_slave(void* args);
int FCNET_backwardPropagation(TwoDMatrix** Ws, TwoDMatrix** Hs, TwoDMatrix** bs, TwoDMatrix** dWs, TwoDMatrix** dbs, TwoDMatrix** dHs, TwoDMatrix* X, TwoDMatrix* dX, int network_depth, float alpha, float reg_strength, int thread_id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
void* FCNET_backwardPropagation_slave(void* args);
int FCNET_updateWeights(TwoDMatrix** Ws, TwoDMatrix** dWs, TwoDMatrix** bs, TwoDMatrix** dbs, TwoDMatrix** Wcaches, TwoDMatrix** bcaches, float learning_rate, float decay_rate,
    float eps, bool use_rmsprop, int network_depth, int thread_id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
void* FCNET_updateWeights_slave(void* args);

void assignSlaveArguments(SlaveArgs* args,
    ThreadControl* handle,
    int thread_id,
    int network_depth,
    TwoDMatrix* X,
    TwoDMatrix** Ws,
    TwoDMatrix** bs,
    TwoDMatrix** Hs,
    TwoDMatrix** dWs,
    TwoDMatrix** dbs,
    TwoDMatrix** dHs,
    TwoDMatrix** Wcaches,
    TwoDMatrix** bcaches,
    TwoDMatrix* correct_labels,
    TwoDMatrix* tmp,
    float alpha,
    float learning_rate,
    float reg_strength,
    float decay_rate,
    float eps,
    bool use_rmsprop,
    bool* mem_allocated,
    int number_of_threads,
    pthread_mutex_t* mutex,
    pthread_cond_t* cond,
    thread_barrier_t* barrier,
    int* int_retval,
    float* float_retval);

#endif
