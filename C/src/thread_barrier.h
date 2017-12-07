#ifndef __THREAD_BARRIER_HEADER__
#define __THREAD_BARRIER_HEADER__

#include <pthread.h>
#ifndef PTHREAD_BARRIER_SERIAL_THREAD
#define PTHREAD_BARRIER_SERIAL_THREAD -1
#endif

#ifndef EINIT_INITIALIZED
#define EINIT_INITIALIZED 1
#endif
#ifndef EINIT_BUSY
#define EINIT_BUSY 2
#endif
#ifndef EDSTRY_UNINIT
#define EDSTRY_UNINIT 3
#endif
#ifndef EDSTRY_BUSY
#define EDSTRY_BUSY 4
#endif

typedef struct {
    pthread_cond_t c;
    pthread_mutex_t m;
    int remain;
    bool busy;
    bool initialized;
} thread_barrier_t;

extern const thread_barrier_t THREAD_BARRIER_INITIALIZER;

int thread_barrier_init(thread_barrier_t* b, int n);
int thread_barrier_destroy(thread_barrier_t* b);
int thread_barrier_wait(thread_barrier_t* b);

#endif
