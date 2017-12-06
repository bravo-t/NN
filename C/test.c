#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <malloc.h>
#include <string.h>
#include <pthread.h>

pthread_mutex_t printf_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_barrier_t barrier;

void signal_threads();

int main() {
    int number_of_threads = 4;
    pthread_barrier_init(&barrier,0,number_of_threads+1);
    int* ids = malloc(sizeof(int)*number_of_threads);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_t* threads = malloc(sizeof(pthread_t)*number_of_threads);
    for(int i=0;i<number_of_threads;i++) {
        ids[i] = i;
        int create_error = pthread_create(&threads[i],&attr,thread,&ids[i]);
        if (create_error) {
            printf("ERROR: Create thread failed.\n");
            exit(-1);
        }
    }
    signal_threads();
    signal_threads();
    void* status;
    for(int i=0;i<number_of_threads;i++) {
        int join_error = pthread_join(threads[i],&status);
        if (join_error) {
            printf("ERROR: Join thread failed.\n");
            exit(-1);
        }
    }
    free(ids);
    return 0;
}

void signal_threads() {
    int r = pthread_barrier_wait(&barrier);
    if (r == PTHREAD_BARRIER_SERIAL_THREAD) pthread_barrier_init(&barrier,0,number_of_threads+1);
}

void* thread(void* id) {
    int i = 0;
    while(1) {
        i++;
        barrier_print(i,(int*) id);
    }
}

void barrier_print(int i, int* id) {
    int r = pthread_barrier_wait(&barrier);
    if (r == PTHREAD_BARRIER_SERIAL_THREAD) pthread_barrier_init(&barrier,0,number_of_threads+1);
    pthread_mutex_lock(&printf_mutex);
    printf("Thread %d: i = %d\n",*id, i);
    pthread_mutex_unlock(&printf_mutex);
}
