#ifndef __THREAD_CONTROL_HEADER__
#define __THREAD_CONTROL_HEADER__

#define THREAD_RESUME 0
#define THREAD_EXIT 1

#define CONTROL_WAIT_INST 0
#define CONTROL_EXEC_COMPLETE 1

typedef struct {
    pthread_mutex_t* mutex;
    thread_barrier_t* inst_ready;
    thread_barrier_t* inst_ack;
    thread_barrier_t* thread_complete;
    int inst;
    int number_of_threads;
} ThreadControl;

void threadController_slave(ThreadControl* handle, int state);
void threadController_master(ThreadControl* handle, int inst_id);
ThreadControl* initControlHandle(pthread_mutex_t* mutex, thread_barrier_t* rdy, thread_barrier_t* ack, thread_barrier_t* complete, int number_of_threads);


#endif
