#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include "src/inter-process_communication.h"

#define THREAD_RESUME 0
#define THREAD_EXIT 1

typedef struct {
    pthread_mutex_t* mutex;
    pthread_cond_t* cond;
    sem_t* semaphore;
    bool cond_set;
    int state;
    int iterations;
} ThreadControl;

typedef struct {
    ThreadControl* handle;
    int id;
} TestArgs;

void* test(void* a);
void threadController_slave(ThreadControl* handle, int id);
ThreadControl* initControlHandle(pthread_mutex_t* mutex, pthread_cond_t* cond, sem_t* sem);
void threadController_master(ThreadControl* handle, int state_id, int number_of_threads);
void waitUntilEveryoneIsFinished_test(sem_t *sem);

pthread_mutex_t printf_mutex = PTHREAD_MUTEX_INITIALIZER;

pthread_mutex_t test_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t test_cond = PTHREAD_COND_INITIALIZER;
//state_t test_state = RESUME;
bool test_set = false;
int main() {
    int number_of_threads = 8;
    sem_t sem;
    ThreadControl* control_handle = initControlHandle(&test_mutex, &test_cond, &sem);
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
    threadController_master(control_handle, THREAD_RESUME, number_of_threads);
    pthread_mutex_lock(&printf_mutex);
    printf("Signal all threads to exit\n");
    pthread_mutex_unlock(&printf_mutex);
    threadController_master(control_handle, THREAD_EXIT, number_of_threads);
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
    static __thread int i = 0;
    if (i == 0) {
        pthread_mutex_lock(&printf_mutex);
        printf("Thread %d: Waiting for instructions...\n",id);
        pthread_mutex_unlock(&printf_mutex);
        int sem_value;
        sem_getvalue(handle->semaphore, &sem_value);
        pthread_mutex_lock(&printf_mutex);
        printf("Thread %d: semaphore value: %d\n",sem_value);
        pthread_mutex_unlock(&printf_mutex);
        sem_wait(handle->semaphore);
        pthread_mutex_lock(&printf_mutex);
        printf("Thread %d: Instructions received...\n",id);
        pthread_mutex_unlock(&printf_mutex);
        i = handle->iterations;
    }
    i--;
    pthread_mutex_lock(handle->mutex);
    while(!(handle->cond_set)) pthread_cond_wait(handle->cond,handle->mutex);
    pthread_mutex_unlock(handle->mutex);
    if (handle->state == THREAD_EXIT) {
        pthread_mutex_lock(&printf_mutex);
        printf("Thread %d: Exiting...\n",id);
        pthread_mutex_unlock(&printf_mutex);
        pthread_exit(NULL);
    } else if (handle->state == THREAD_RESUME) {
        pthread_mutex_lock(&printf_mutex);
        printf("Thread %d: Resuming...\n",id);
        pthread_mutex_unlock(&printf_mutex);
        return;
    }
}

ThreadControl* initControlHandle(pthread_mutex_t* mutex, pthread_cond_t* cond, sem_t* sem) {
    ThreadControl* handle = malloc(sizeof(ThreadControl));
    handle->cond_set = false;
    handle->mutex = mutex;
    handle->cond = cond;
    // Initialize the semaphore to 0 so that all slave thread will
    // hold their actions and wait for instructions
    sem_init(sem,0,0);
    handle->semaphore = sem;
    handle->state = THREAD_RESUME;
    handle->iterations = 1;
    return handle;
}

void threadController_master(ThreadControl* handle, int state_id, int number_of_threads) {
    pthread_mutex_lock(handle->mutex);
    handle->state = state_id;
    handle->cond_set = true;
    for(int i=0;i<number_of_threads;i++) sem_post(handle->semaphore);
    pthread_cond_broadcast(handle->cond);
    pthread_mutex_unlock(handle->mutex);
    pthread_mutex_lock(&printf_mutex);
    printf("Wait threads to finish\n");
    pthread_mutex_unlock(&printf_mutex);
    //waitUntilEveryoneIsFinished_test(handle->semaphore);
    pthread_mutex_lock(&printf_mutex);
    printf("Thread finished\n");
    pthread_mutex_unlock(&printf_mutex);
}

void waitUntilEveryoneIsFinished_test(sem_t *sem) {
    while (sem_trywait(sem) != -1 && errno != EAGAIN) {
        sem_post(sem);
        pthread_mutex_lock(&printf_mutex);
        printf("Thread is still running\n");
        pthread_mutex_unlock(&printf_mutex);
    }
}