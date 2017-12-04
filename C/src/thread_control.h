#ifndef __THREAD_CONTROL_HEADER__
#define __THREAD_CONTROL_HEADER__

typedef enum {RESUME,EXIT} state_t;

typedef struct {
    pthread_mutex_t* mutex;
    pthread_cond_t* cond;
    bool* cond_set;
    state_t* state;
} ThreadControl;

#endif
