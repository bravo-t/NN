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
    bool* memory_allocated;
    float alpha;
    float learning_rate;
    float reg_strength;
    float decay_rate;
    float eps;
} SlaveArgs;

#endif
