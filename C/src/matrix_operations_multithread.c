#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>
#include "network_type.h"
#include "inter-process_communication.h"
#include "matrix_operations.h"
#include "matrix_operations_multithread.h"

extern pthread_mutex_t mutex;
extern pthread_cond_t cond;
extern sem_t semaphore;
extern pthread_barrier_t barrier;

int init2DMatrix_MT(TwoDMatrix* M, int height, int width, int h_start, int h_end, bool* mem_allocated) {
    if (M->initialized) return 0;
    if (h_start == 0) {
    	pthread_mutex_lock(&mutex);
    	M->height = height;
    	M->width = width;
    	M->d = (float**) calloc(height, sizeof(float*));
    	*mem_allocated = true;
    	pthread_mutex_unlock(&mutex);
    } else {
    	pthread_mutex_lock(&mutex);
    	while(!(*mem_allocated)) pthread_cond_wait(&cond);
    	pthread_mutex_unlock(&mutex);
    }
    if(h_start == 0) {
    	pthread_mutex_lock(&mutex);
    	pthread_cond_broadcast(&cond);
    	pthread_mutex_unlock(&mutex);
    }
    for(int i=h_start;i<=h_end;i++) {
        M->d[i] = (float*) calloc(width,sizeof(float));
    }
    if(h_start == 0) M->initialized = true;
    pthead_barrier_wait(&barrier);
    return 0;
}

int init2DMatrixNormRand_MT(TwoDMatrix* M, int height, int width, float mean, float std, int n,int h_start, int h_end, bool* mem_allocated) {
	if (M->initialized) return 0;
    if (h_start == 0) {
    	pthread_mutex_lock(&mutex);
    	M->height = height;
    	M->width = width;
    	M->d = (float**) calloc(height, sizeof(float*));
    	*mem_allocated = true;
    	pthread_mutex_unlock(&mutex);
    } else {
    	pthread_mutex_lock(&mutex);
    	while(!(*mem_allocated)) pthread_cond_wait(&cond);
    	pthread_mutex_unlock(&mutex);
    }
    if(h_start == 0) {
    	pthread_mutex_lock(&mutex);
    	pthread_cond_broadcast(&cond);
    	pthread_mutex_unlock(&mutex);
    }
    for(int i=h_start;i<=h_end;i++) {
        M->d[i] = (float*) calloc(width,sizeof(float));
        for(int j=0;j<width;j++) {
            data[i][j] = random_normal(mean,std)*sqrt(2.0/n);
        }
    }
    if(h_start == 0) M->initialized = true;
    pthead_barrier_wait(&barrier);
    return 0;
}

int init2DMatrixZero_MT(TwoDMatrix* M, int height, int width,int h_start, int h_end, bool* mem_allocated) {
    init2DMatrix(M,height,width,h_start,h_end,mem_allocated);
    return 0;
}

int init2DMatrixOne(TwoDMatrix* M, int height, int width,int h_start, int h_end, bool* mem_allocated) {
    if (M->initialized) return 0;
    if (h_start == 0) {
    	pthread_mutex_lock(&mutex);
    	M->height = height;
    	M->width = width;
    	M->d = (float**) calloc(height, sizeof(float*));
    	*mem_allocated = true;
    	pthread_mutex_unlock(&mutex);
    } else {
    	pthread_mutex_lock(&mutex);
    	while(!(*mem_allocated)) pthread_cond_wait(&cond);
    	pthread_mutex_unlock(&mutex);
    }
    if(h_start == 0) {
    	pthread_mutex_lock(&mutex);
    	pthread_cond_broadcast(&cond);
    	pthread_mutex_unlock(&mutex);
    }
    for(int i=h_start;i<=h_end;i++) {
        M->d[i] = (float*) calloc(width,sizeof(float));
        for(int j=0;j<width;j++) {
            data[i][j] = 1;
        }
    }
    if(h_start == 0) M->initialized = true;
    pthead_barrier_wait(&barrier);
    return 0;
}