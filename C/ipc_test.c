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
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/mman.h>
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
    char* shared_memory_name = "/ipc_test";
    TwoDMatrixOperationsArgs* a = (TwoDMatrixOperationsArgs*) args;
    if (a->h_start != 0) sleep(10);
    if (a->h_start == 0) {
        pthread_mutex_lock(&mutex);
        printf("Thread %d: allocating memory for output 2D matrix\n", a->t);
        a->OUT = matrixMalloc(sizeof(TwoDMatrix));
        int shm_fd = shm_open(shared_memory_name,O_CREAT|O_RDWR);
        if (shm_fd == -1) {
            printf("ERROR: Cannot create shared memory\n");
            exit(1);
        }
        ftruncate(shm_fd,sizeof(TwoDMatrix*));
        int* shm_base = mmap(0,sizeof(TwoDMatrix*),PROT_READ|PROT_WRITE,MAP_SHARED,shm_fd,0);
        if (shm_base == MAP_FAILED) {
            printf("ERROR: mmap failed\n");
            exit(1);
        }
        *shm_base = a->OUT;
        printf("Wrote 0x%x to shared memory\n",a->OUT);
        //init2DMatrix(a->OUT,a->height,a->width);
        *(a->memory_allocated) = true;
        pthread_mutex_unlock(&mutex);
    } else {
        // The mutex is a must to prevent a race condition
        // without the mutex, signals can arrive when threads finished checking a->memory_allocated and before entering pthead_cond_wait
        // The wake up signal will be lost for a situation like this
        pthread_mutex_lock(&mutex);
        // The while loop is for detecting spurious wake-ups
        while (! *(a->memory_allocated)) {
            printf("Thread %d: wait for thread 0 to allocate memory\n",a->t);
            // NOTE: pthread_cond_wait will only wake up when cond is not signaled when pthread_cond_wait is entered
            // In other words, signals before entering pthread_cond_wait are ignored and the thread will be sleeping forever
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
