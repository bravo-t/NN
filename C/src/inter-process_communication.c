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
#include <errno.h>
#include <semaphore.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include "network_type.h"

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
sem_t semaphore;
pthread_barrier_t barrier;

void microsecSleep (long ms) {
    struct timeval delay;
    delay.tv_sec = ms * 1e-6;
    delay.tv_usec = ms - 1e6*delay.tv_sec;
    select(0,NULL,NULL,NULL,&delay);
}

void waitUntilEveryoneIsFinished(sem_t *sem) {
    while (sem_trywait(sem) != -1 && errno != EAGAIN) {
        sem_post(sem);
        //microsec_sleep(1);
    }
}

int IPCWriteToSharedMem(char* shared_memory_name, void* base_ptr, int data_length) {
	int shm_fd = shm_open(shared_memory_name,O_CREAT|O_RDWR,0666);
    if (shm_fd == -1) {
        printf("ERROR: Cannot create shared memory\n");
        return 1;
    }
    ftruncate(shm_fd,data_length);
    void** shm_base = mmap(0,data_length,PROT_READ|PROT_WRITE,MAP_SHARED,shm_fd,0);
    if (shm_base == MAP_FAILED) {
        printf("ERROR: mmap failed\n");
        return 1;
    }
    memcpy(*shm_base, base_ptr, data_length);
    if (munmap(shm_base,data_length)) {
        printf("Unmap failed\n");
        return 1;
    }
    if (close(shm_fd) == -1) {
        printf("Close failed\n");
        return 1;
    }
    return 0;
}

int IPCReadFromSharedMem(char* shared_memory_name, void* data, int data_length) {
	int shm_fd = shm_open(shared_memory_name,O_RDONLY,0666);
    if (shm_fd == -1) {
        printf("ERROR: Cannot read shared memory\n");
        return(1);
    }
    TwoDMatrix** shm_base = mmap(0,data_length,PROT_READ,MAP_SHARED,shm_fd,0);
    if (shm_base == MAP_FAILED) {
        printf("mmap failed\n");
        return(1);
    }
    memcpy(data, *shm_base, data_length);
    if (munmap(shm_base,sizeof(TwoDMatrix))) {
        printf("Unmap failed\n");
        return(1);
    }
    if (close(shm_fd) == -1) {
        printf("Close failed\n");
        return(1);
    }
    return 0;
}

int IPCRemoveSharedMemFile(char* shared_memory_name) {
	if (shm_unlink(shared_memory_name) == -1) {
        printf("Error occured when removing shared memory file\n");
        return 1;
    }
    return 0;
}
