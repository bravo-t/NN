#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>
#include "network_type.h"
#include "matrix_operations.h"
#include "matrix_operations_multithread.h"

int dotProduct_MT(TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* OUT, int number_of_threads) {
    if (X->width != W->height) {
        return 1;
    }
    init2DMatrix(OUT,X->height,W->width);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_t* thread = malloc(sizeof(pthread_t)*number_of_threads);
    int H = X->height / number_of_threads + 1;
    int t = 0;
    for(;t<number_of_threads;t++) {
        struct DotProductRowArgs* thread_arg = malloc(sizeof(struct DotProductRowArgs));
        thread_arg->X = X;
        thread_arg->W = W;
        thread_arg->OUT = OUT;
        thread_arg->h_start = t*H;
        if (thread_arg->h_start >= X->height) break;
        thread_arg->h_end = (t+1)*H-1;
        if (thread_arg->h_end >= X->height) thread_arg->h_end = X->height - 1;
        int create_error = pthread_create(&thread[t],&attr,dotProductRow,(void*) thread_arg);
        if (create_error) {
            printf("ERROR: Create thread failed.\n");
            exit(-1);
        }
    }
    void* status;
    for(int n=0;n<t;n++) {
        int join_error = pthread_join(thread[n],&status);
        if (join_error) {
            printf("ERROR: Join thread failed.\n");
            exit(-1);
        }
    }        
    return 0;
}

void* dotProductRow(void* args) {
    struct DotProductRowArgs* a = (struct DotProductRowArgs*) args;
    TwoDMatrix* X = a->X;
    TwoDMatrix* W = a->W;
    TwoDMatrix* OUT = a->OUT;
    for(int i=a->h_start;i<=a->h_end;i++) {
        for(int j=0;j<W->width;j++) {
            float sum = 0;
            for(int p=0;p<X->width;p++) sum += X->d[i][p]*W->d[p][j];
            OUT->d[i][j] = sum;
        }
    }
    free(args);
    pthread_exit(NULL);
}