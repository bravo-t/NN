#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <malloc.h>
#include <string.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>
#include "src/thread_barrier.h"
#include "src/inter-process_communication.h"
#include "src/network_type.h"
#include "src/matrix_operations.h"
#include "src/matrix_operations_multithread.h"

void* thread(void* id);
bool mem_allocated = false;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
thread_barrier_t* barrier;

int number_of_threads = 4;

int main() {
	*barrier = THREAD_BARRIER_INITIALIZER;
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

void* thread(void* id) {

	int* thread_id = (int*) id;
	char* test = malloc_thread("/malloc_thread_test",sizeof(char)*10,*thread_id,&mem_allocated,number_of_threads,&mutex,&cond,barrier);
	printf("DEBUG: id %d: test = %p\n", *thread_id, test);
}
