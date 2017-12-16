#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <pthread.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include "thread_barrier.h"
#include "thread_control.h"

void threadController_slave(ThreadControl* handle) {
    int instruction;
    thread_barrier_wait_reinit(handle->inst_ready,handle->number_of_threads+1);
    instruction = handle->inst;
    thread_barrier_wait_reinit(handle->inst_ack,handle->number_of_threads+1);
    if (instruction == THREAD_EXIT) {
        pthread_exit(NULL);
    } else if (instruction == THREAD_RESUME) {
        return;
    }
}

ThreadControl* initControlHandle(pthread_mutex_t* mutex, thread_barrier_t* rdy, thread_barrier_t* ack, int number_of_threads) {
    ThreadControl* handle = malloc(sizeof(ThreadControl));
    handle->mutex = mutex;
    thread_barrier_init(rdy, number_of_threads+1);
    handle->inst_ready = rdy;
    thread_barrier_init(ack, number_of_threads+1);
    handle->inst_ack = ack;
    handle->inst = THREAD_RESUME;
    handle->number_of_threads = number_of_threads;
    return handle;
}

void threadController_master(ThreadControl* handle, int inst_id) {
    pthread_mutex_lock(handle->mutex);
    handle->inst = inst_id;
    pthread_mutex_unlock(handle->mutex);
    thread_barrier_wait_reinit(handle->inst_ready,handle->number_of_threads+1);
    thread_barrier_wait_reinit(handle->inst_ack,handle->number_of_threads+1);
}


