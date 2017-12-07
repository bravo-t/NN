#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <pthread.h>
#include <errno.h>
#include "thread_barrier.h"


const thread_barrier_t THREAD_BARRIER_INITIALIZER = {
    PTHREAD_COND_INITIALIZER,
    PTHREAD_MUTEX_INITIALIZER,
    0,
    false,
    false,
};

int thread_barrier_init(thread_barrier_t* b, int n) {
    pthread_mutex_lock(&(b->m));
    if (b->initialized) {
        return EINIT_INITIALIZED;
    }
    if (b->busy) {
        return EINIT_BUSY;
    }
    pthread_cond_init(&(b->c),NULL);
    b->remain = n;
    b->initialized = true;
    pthread_mutex_unlock(&(b->m));
    return 0;
}

int thread_barrier_destroy(thread_barrier_t* b) {
    pthread_mutex_lock(&(b->m));
    if (! b->initialized) {
        return EDSTRY_UNINIT;
    }
    if (b->busy) {
        return EDSTRY_BUSY;
    }
    b->remain = 0;
    pthread_cond_destroy(&(b->c));
    b->initialized = false;
    pthread_mutex_unlock(&(b->m));
    return 0;
}

int thread_barrier_wait(thread_barrier_t* b) {
    if (! b->initialized) {
        return EDSTRY_UNINIT;
    }
    pthread_mutex_lock(&(b->m));
    if (! b->busy) b->busy = true;
    b->remain--;
    printf("DEBUG: barrier_wait: b->remain = %d\n",b->remain);
    int retval = 0;
    if (b->remain == 0) {
        pthread_cond_broadcast(&(b->c));
        b->busy = false;
        retval = PTHREAD_BARRIER_SERIAL_THREAD;
    } else {
        while(b->remain != 0) pthread_cond_wait(&(b->c),&(b->m));
    }
    pthread_mutex_unlock(&(b->m));
    return retval;
}
