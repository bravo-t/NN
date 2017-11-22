#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <stdarg.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include "src/network_type.h"
#include "src/matrix_operations.h"
#include "src/layers.h"
#include "src/misc_utils.h"
#include "src/fully_connected_net.h"

typedef struct {
    TwoDMatrix* OUT;
    int height;
    int width;
    int h_start;
    int h_end;
    int t;
    bool* memory_allocated;
} TwoDMatrixOperationsArgs;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
bool memory_allocated = false;

void* testPartial(void* args);

int main() {
    int number_of_threads = 8;
    int height = 10;
    int width = 5;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_t* thread = malloc(sizeof(pthread_t)*number_of_threads);
    int H = height / number_of_threads + 1;
    int t = 0;
    bool memory_allocated = false;
    for(;t<number_of_threads;t++) {
        TwoDMatrixOperationsArgs* thread_arg = malloc(sizeof(TwoDMatrixOperationsArgs));
        thread_arg->h_start = t*H;
        if (thread_arg->h_start >= height) break;
        thread_arg->h_end = (t+1)*H-1;
        if (thread_arg->h_end >= height) thread_arg->h_end = height - 1;
        thread_arg->height = height;
        thread_arg->width = width;
        thread_arg->t = t;
        thread_arg->memory_allocated = &memory_allocated;
        pthread_mutex_lock(&mutex);
        printf("Main: dispatching thread %d for %d to %d\n",t,thread_arg->h_start,thread_arg->h_end);
        pthread_mutex_unlock(&mutex);
        int create_error = pthread_create(&thread[t],&attr,testPartial,(void*) thread_arg);
        if (create_error) {
            printf("ERROR: Create thread failed.\n");
            exit(-1);
        }
    }
    void* status;
    for(int n=0;n<t;n++) {
        int join_error = pthread_join(thread[n],&status);
        if (join_error) {
            printf("ERROR: Join thread failed.\n");
            exit(-1);
        }
    } 
    pthread_attr_destroy(&attr);
    free(thread);
    return 0;
}

void* testPartial(void* args) {
    TwoDMatrixOperationsArgs* a = (TwoDMatrixOperationsArgs*) args;
    if (a->h_start == 0) {
        pthread_mutex_lock(&mutex);
        printf("Thread %d: allocating memory for output 2D matrix\n", a->t);
        a->OUT = matrixMalloc(sizeof(TwoDMatrix));
        init2DMatrix(a->OUT,a->height,a->width);
        *(a->memory_allocated) = true;
        pthread_mutex_unlock(&mutex);
    } else {
        pthread_mutex_lock(&mutex);
        while (! *(a->memory_allocated)) {
        printf("Thread %d: wait for thread 0 to allocate memory\n",a->t);
        pthread_cond_wait(&cond,&mutex);
        }
        pthread_mutex_unlock(&mutex);
    }
    if (a->h_start == 0) {
        //sleep(1);
        pthread_mutex_lock(&mutex);
        printf("Thread %d: memory allocated, signal all other threads to continue working\n", a->t);
        pthread_cond_broadcast(&cond);
        pthread_mutex_unlock(&mutex);
    }
    pthread_mutex_lock(&mutex);
    printf("Thread %d: assigning values from %d to %d\n",a->t ,a->h_start,a->h_end);
    pthread_mutex_unlock(&mutex);
    free(args);
    pthread_exit(NULL);
}
