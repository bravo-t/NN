#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>

typedef enum {RESUME,EXIT} state_t;

typedef struct {
    pthread_mutex_t* mutex;
    pthread_cond_t* cond;
    bool* cond_set;
    state_t* state;
} ThreadControl;

typedef struct {
    ThreadControl* handle;
    int id;
} TestArgs;

pthread_mutex_t printf_mutex = PTHREAD_MUTEX_INITIALIZER;

ThreadControl control_handle;
pthread_mutex_t test_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t test_cond = PTHREAD_COND_INITIALIZER;
state_t test_state = state_t.RESUME;
bool test_set = false;
int main() {
    int number_of_threads = 8;
    control_handle.mutex = &test_mutex;
    control_handle.cond = &test_cond;
    control_handle.state = &test_state;
    control_handle.cond_set = &test_set;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_t* thread = malloc(sizeof(pthread_t)*number_of_threads);
    int t = 0;
    for(;t<number_of_threads;t++) {
        TestArgs* a = malloc(sizeof(TestArgs));
        a->handle = (ThreadControl*) malloc(sizeof((ThreadControl)));
        a->handle = &control_handle;
        a->id = t;
        int create_error = pthread_create(&thread[t],&attr,test,a);
        if (create_error) {
            printf("ERROR: Create thread failed.\n");
            exit(-1);
        }
    }
    sleep(1);
    threadController_master(&control_handle, state_t.RESUME);
    sleep(1);
    threadController_master(&control_handle, state_t.RESUME);
    sleep(10);
    threadController_master(&control_handle, state_t.EXIT);
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

void test(void* a) {
    int id = *(a)->id;
    ThreadControl* control_handle = a->handle;
    while(1) {
        pthread_mutex_lock(&printf_mutex);
        printf("Thread %d: ",id);
        pthread_mutex_unlock(&printf_mutex);
        threadController_slave(control_handle);
    }
}


void threadController_slave(ThreadControl* handle) {
    pthread_mutex_lock(handle->mutex);
    while(!(*(handle->cond_set))) pthread_cond_wait(handle->cond,handle->mutex);
    pthread_mutex_unlock(handle->mutex);
    if (*(handle->state) == EXIT) {
        pthread_mutex_lock(&printf_mutex);
        printf("Exiting...\n");
        pthread_mutex_unlock(&printf_mutex);
        pthread_exit(NULL);
    } else if (*(handle->state) == RESUME) {
        pthread_mutex_lock(&printf_mutex);
        printf("Resuming...\n");
        pthread_mutex_unlock(&printf_mutex);
        return;
    }
}

void threadController_master(ThreadControl* handle, int state_id) {
    pthread_mutex_lock(handle->mutex);
    *(handle->state) = state_id;
    pthread_cond_broadcast(handle->cond);
    pthread_mutex_unlock(handle->mutex);
}

