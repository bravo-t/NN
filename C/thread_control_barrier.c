#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <pthread.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include "src/thread_barrier.h"

#define THREAD_RESUME 0
#define THREAD_EXIT 1

typedef struct {
    pthread_mutex_t* mutex;
    thread_barrier_t* inst_ready;
    pthread_barrier_t* inst_ack;
    int inst;
    int number_of_threads;
} ThreadControl;

typedef struct {
    ThreadControl* handle;
    int id;
} TestArgs;

void* test(void* a);
void threadController_slave(ThreadControl* handle, int id);
void threadController_master(ThreadControl* handle, int inst_id);
ThreadControl* initControlHandle(pthread_mutex_t* mutex, thread_barrier_t* rdy, thread_barrier_t* ack, int number_of_threads);
void microsecSleep (long ms);

pthread_mutex_t test_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t printf_mutex = PTHREAD_MUTEX_INITIALIZER;

//state_t test_state = RESUME;
bool test_set = false;
int main() {
    int number_of_threads = 4;
    thread_barrier_t instruction_ready = THREAD_BARRIER_INITIALIZER;
    pthread_barrier_t acknowledge = THREAD_BARRIER_INITIALIZER;
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
        threadController_slave(control_handle,id);
        sleep(1);
    }
}


void threadController_slave(ThreadControl* handle,int id) {
    int instruction;
    pthread_mutex_lock(&printf_mutex);
    printf("Thread %d: Waiting for instructions...\n",id);
    pthread_mutex_unlock(&printf_mutex);
    //int sem_value;
    //sem_getvalue(handle->semaphore, &sem_value);
    //pthread_mutex_lock(&printf_mutex);
    //printf("Thread %d: semaphore value: %d\n",id, sem_value);
    //pthread_mutex_unlock(&printf_mutex);
    
    int r = thread_barrier_wait_reinit(handle->inst_ready,handle->number_of_threads+1);
    instruction = handle->inst;
    // microsecSleep(10);
    //pthread_mutex_lock(handle->mutex);
    /*
    if (r == PTHREAD_BARRIER_SERIAL_THREAD) {
        pthread_mutex_lock(&printf_mutex);
        printf("Thread %d: re-init inst_ready barrier\n",id);
        pthread_mutex_unlock(&printf_mutex);
     //   microsecSleep(1);
        int d;
        do { d = thread_barrier_destroy(handle->inst_ready); } while(d == THREAD_BARRIER_EDSTRY_BUSY);
        int ie;
        do { ie = thread_barrier_init(handle->inst_ready,(handle->number_of_threads)+1);} while (ie == THREAD_BARRIER_EINIT_BUSY);
    }
    */
    //pthread_mutex_unlock(handle->mutex);
    pthread_mutex_lock(&printf_mutex);
    printf("Thread %d: Instructions received...\n",id);
    pthread_mutex_unlock(&printf_mutex);
    //pthread_mutex_lock(handle->mutex);
    //while(!(handle->cond_set)) pthread_cond_wait(handle->cond,handle->mutex);
    //pthread_mutex_unlock(handle->mutex);
    int a = pthread_barrier_wait_reinit(handle->inst_ack,handle->number_of_threads+1);
    //pthread_mutex_lock(handle->mutex);
    //if (e == PTHREAD_BARRIER_SERIAL_THREAD) {
    //    int d;
    //    do { d = pthread_barrier_destroy(handle->exec_inst); } while(d != 0);
    //    pthread_barrier_init(handle->exec_inst,0,(handle->number_of_threads)+1);
    //}
    //pthread_mutex_unlock(handle->mutex);
    if (instruction == THREAD_EXIT) {
        pthread_mutex_lock(&printf_mutex);
        printf("Thread %d: Exiting...\n",id);
        pthread_mutex_unlock(&printf_mutex);
        pthread_exit(NULL);
    } else if (instruction == THREAD_RESUME) {
        pthread_mutex_lock(&printf_mutex);
        printf("Thread %d: Resuming...\n",id);
        pthread_mutex_unlock(&printf_mutex);
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
    int r = thread_barrier_wait_reinit(handle->inst_ready,handle->number_of_threads+1);
    //pthread_mutex_lock(handle->mutex);
    /*
    if (r == PTHREAD_BARRIER_SERIAL_THREAD) {
        pthread_mutex_lock(&printf_mutex);
        printf("Main: re-init inst_ready barrier\n");
        pthread_mutex_unlock(&printf_mutex);
        //microsecSleep(10);
        int d;
        do { d = thread_barrier_destroy(handle->inst_ready); } while(d == THREAD_BARRIER_EDSTRY_BUSY);
        int ie;
        do {thread_barrier_init(handle->inst_ready,(handle->number_of_threads)+1);} while(ie == THREAD_BARRIER_EINIT_BUSY);
    }
    */
    //pthread_mutex_unlock(handle->mutex);
    pthread_mutex_lock(&printf_mutex);
    printf("Signaled %d threads, wait for them to finish\n",handle->number_of_threads);
    pthread_mutex_unlock(&printf_mutex);
    int a = pthread_barrier_wait(handle->inst_ack,handle->number_of_threads+1);
    //pthread_mutex_lock(handle->mutex);
    //if (e == PTHREAD_BARRIER_SERIAL_THREAD) {
    //    int d;
    //    do { d = pthread_barrier_destroy(handle->exec_inst); } while(d != 0);
    //    pthread_barrier_init(handle->exec_inst,0,(handle->number_of_threads)+1);
    //} 
    //pthread_mutex_unlock(handle->mutex);
    pthread_mutex_lock(&printf_mutex);
    printf("Thread finished\n");
    pthread_mutex_unlock(&printf_mutex);
}

void microsecSleep (long ms) {
    struct timeval delay;
    delay.tv_sec = ms * 1e-6;
    delay.tv_usec = ms - 1e6*delay.tv_sec;
    select(0,NULL,NULL,NULL,&delay);
}

