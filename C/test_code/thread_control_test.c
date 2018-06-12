#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <pthread.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include "src/thread_barrier.h"
#include "src/thread_control.h"

typedef struct {
    ThreadControl* handle;
    int id;
} TestArgs;

void* test(void* a);
void microsecSleep (long ms);

pthread_mutex_t test_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t printf_mutex = PTHREAD_MUTEX_INITIALIZER;

//state_t test_state = RESUME;
bool test_set = false;
int main() {
    int number_of_threads = 4;
    thread_barrier_t instruction_ready = THREAD_BARRIER_INITIALIZER;
    thread_barrier_t acknowledge = THREAD_BARRIER_INITIALIZER;
    ThreadControl* control_handle = initControlHandle(&test_mutex, &instruction_ready, &acknowledge, number_of_threads);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_t* thread = malloc(sizeof(pthread_t)*number_of_threads);
    int t = 0;
    for(;t<number_of_threads;t++) {
        TestArgs* a = malloc(sizeof(TestArgs));
        a->handle = (ThreadControl*) malloc(sizeof(ThreadControl));
        a->handle = control_handle;
        a->id = t;
        int create_error = pthread_create(&thread[t],&attr,test,a);
        if (create_error) {
            printf("ERROR: Create thread failed.\n");
            exit(-1);
        }
    }
    pthread_mutex_lock(&printf_mutex);
    printf("Signal all threads to resume\n");
    pthread_mutex_unlock(&printf_mutex);
    threadController_master(control_handle, THREAD_RESUME);
    pthread_mutex_lock(&printf_mutex);
    printf("Signal all threads to exit\n");
    pthread_mutex_unlock(&printf_mutex);
    threadController_master(control_handle, THREAD_EXIT);
    void* status;
    for(int n=0;n<t;n++) {
        int join_error = pthread_join(thread[n],&status);
        if (join_error) {
            printf("ERROR: Join thread failed.\n");
            exit(-1);
        }
    }
    pthread_attr_destroy(&attr);


    return 0;
}

void* test(void* a) {
    TestArgs* args = (TestArgs*) a;
    int id = (*args).id;
    ThreadControl* control_handle = (*args).handle;
    while(1) {
        threadController_slave(control_handle);
        pthread_mutex_lock(&printf_mutex);
        printf("Thread %d resuming\n", id);
        pthread_mutex_unlock(&printf_mutex);
        sleep(1);
    }
}


void microsecSleep (long ms) {
    struct timeval delay;
    delay.tv_sec = ms * 1e-6;
    delay.tv_usec = ms - 1e6*delay.tv_sec;
    select(0,NULL,NULL,NULL,&delay);
}
