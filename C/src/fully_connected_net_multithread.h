#ifndef __FULLY_CONNECTED_MT_HEADER__
#define __FULLY_CONNECTED_MT_HEADER__

#include "thread_barrier.h"
#include "thread_control.h"

typedef struct {
    ThreadControl* handle;
    int network_depth;
    TwoDMatrix* X;
    TwoDMatrix** Ws;
    TwoDMatrix** bs;
    TwoDMatrix** Hs;
    TwoDMatrix** dWs;
    TwoDMatrix** dbs;
    TwoDMatrix** dHs;
    float alpha;
    float learning_rate;

} SlaveArgs;

#endif
