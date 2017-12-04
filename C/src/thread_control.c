#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

void threadController_slave(ThreadControl* control) {
    pthread_mutex_lock(&mutex);
    while(!(*mem_allocated)) pthread_cond_wait(&cond,&mutex);
    pthread_mutex_unlock(&mutex);
}

void threadController_master(ThreadControl* control) {
    pthread_mutex_lock(&mutex);
    pthread_cond_broadcast(&cond);

    pthread_mutex_unlock(&mutex);
}
