#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <stdarg.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <semaphore.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/mman.h>

void threadController_slave(ThreadControl* handle) {
    pthread_mutex_lock(handle->mutex);
    while(!(*(handle->cond_set))) pthread_cond_wait(handle->cond,handle->mutex);
    pthread_mutex_unlock(handle->mutex);
    if (*(handle->state) == EXIT) {
        pthread_exit(NULL);
    } else if (*(handle->state) == RESUME) {
        return;
    }
}

void threadController_master(ThreadControl* handle, int state_id) {
    pthread_mutex_lock(handle->mutex);
    *(handle->state) = state_id;
    pthread_cond_broadcast(handle->cond);
    pthread_mutex_unlock(handle->mutex);
}


