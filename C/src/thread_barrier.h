#ifndef __THREAD_BARRIER_HEADER__
#define __THREAD_BARRIER_HEADER__

#include <pthread.h>
#ifndef PTHREAD_BARRIER_SERIAL_THREAD
#define PTHREAD_BARRIER_SERIAL_THREAD -1
#endif

#ifndef THREAD_BARRIER_EINIT_INITIALIZED
#define THREAD_BARRIER_EINIT_INITIALIZED 1
#endif
#ifndef THREAD_BARRIER_EINIT_BUSY
#define THREAD_BARRIER_EINIT_BUSY 2
#endif
#ifndef THREAD_BARRIER_EDSTRY_UNINIT
#define THREAD_BARRIER_EDSTRY_UNINIT 3
#endif
#ifndef THREAD_BARRIER_EDSTRY_BUSY
#define THREAD_BARRIER_EDSTRY_BUSY 4
#endif
#ifndef THREAD_BARRIER_EDSTRY_NOTYET
#define THREAD_BARRIER_EDSTRY_NOTYET 5
#endif

typedef struct {
    pthread_cond_t c;
    pthread_mutex_t m;
    int remain;
    int total;
    int released;
    bool to_be_destroyed;
    bool initialized;
} thread_barrier_t;

extern const thread_barrier_t THREAD_BARRIER_INITIALIZER;

int thread_barrier_init(thread_barrier_t* b, int n);
int thread_barrier_destroy(thread_barrier_t* b);
int thread_barrier_wait(thread_barrier_t* b);
int thread_barrier_wait_reinit(thread_barrier_t* b, int n);

#endif
