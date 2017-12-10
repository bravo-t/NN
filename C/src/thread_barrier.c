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
    0,
    0,
    false,
    false,
};

int thread_barrier_init(thread_barrier_t* b, int n) {
    if (b->initialized) return THREAD_BARRIER_EINIT_INITIALIZED;
    if (b->released != b->total) return THREAD_BARRIER_EINIT_BUSY;
    pthread_mutex_lock(&(b->m));
    pthread_cond_init(&(b->c),NULL);
    b->remain = n;
    b->total = n;
    b->released = 0;
    b->initialized = true;
    pthread_mutex_unlock(&(b->m));
    return 0;
}

int thread_barrier_destroy(thread_barrier_t* b) {
    if (! b->initialized) return THREAD_BARRIER_EDSTRY_UNINIT;
    if (b->released != b->total) return THREAD_BARRIER_EDSTRY_BUSY;
    if (! b->to_be_destroyed) return THREAD_BARRIER_EDSTRY_NOTYET;
    pthread_mutex_lock(&(b->m));
    b->remain = 0;
    pthread_cond_destroy(&(b->c));
    b->released = b->total;
    b->to_be_destroyed = false;
    b->initialized = false;
    pthread_mutex_unlock(&(b->m));
    return 0;
}

int thread_barrier_wait(thread_barrier_t* b) {
    bool wait_until_reinit;
    do {
        pthread_mutex_lock(&(b->m));
        wait_until_reinit = b->to_be_destroyed || (! b->initialized);
        pthread_mutex_unlock(&(b->m));
    } while (wait_until_reinit);
    pthread_mutex_lock(&(b->m));
    b->remain--;
    printf("DEBUG: barrier_wait: b->remain = %d\n",b->remain);
    int retval = 0;
    if (b->remain == 0) {
        b->to_be_destroyed = true;
        pthread_cond_broadcast(&(b->c));
        retval = PTHREAD_BARRIER_SERIAL_THREAD;
    } else {
        while(b->remain != 0) pthread_cond_wait(&(b->c),&(b->m));
    }
    b->released++;
    pthread_mutex_unlock(&(b->m));
    return retval;
}
