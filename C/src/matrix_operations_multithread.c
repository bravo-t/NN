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

int init2DMatrix_MT(TwoDMatrix* M, int height, int width, int number_of_threads) {
    if (M->initialized) return 0;
    M->height = height;
    M->width = width;
    float** data = (float**) calloc(height, sizeof(float*));
    M->d = data;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_t* thread = malloc(sizeof(pthread_t)*number_of_threads);
    int H = height / number_of_threads + 1;
    int t = 0;
    for(;t<number_of_threads;t++) {
        struct TwoMatrixOperationsRowArgs* thread_arg = malloc(sizeof(struct TwoMatrixOperationsRowArgs));
        thread_arg->M = M;
        thread_arg->h_start = t*H;
        if (thread_arg->h_start >= height) break;
        thread_arg->h_end = (t+1)*H-1;
        if (thread_arg->h_end >= height) thread_arg->h_end = height - 1;
        int create_error = pthread_create(&thread[t],&attr,init2DMatrixRow,(void*) thread_arg);
        if (create_error) {
            printf("ERROR: Create thread failed.\n");
            exit(-1);
        }
    void* status;
    for(int n=0;n<t;n++) {
        int join_error = pthread_join(thread[n],&status);
        if (join_error) {
            printf("ERROR: Join thread failed.\n");
            exit(-1);
        }
    } 
    pthread_attr_destroy(&attr);
    M->initialized = true;
    return 0;
}

void* init2DMatrixRow(void* args) {
    struct TwoMatrixOperationsRowArgs* a = (struct TwoMatrixOperationsRowArgs*) args;
    TwoDMatrix* M = a->M;
    for(int i=a->h_start;i<=a->h_end;i++) {
        M->d[i] = (float*) calloc(M->width,sizeof(float));
    }
    free(args);
    pthread_exit(NULL);
}

int init2DMatrixZero_MT(TwoDMatrix* M, int height, int width, int number_of_threads) {
    if (M->initialized) return 0;
    M->height = height;
    M->width = width;
    float** data = (float**) calloc(height, sizeof(float*));
    M->d = data;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_t* thread = malloc(sizeof(pthread_t)*number_of_threads);
    int H = height / number_of_threads + 1;
    int t = 0;
    for(;t<number_of_threads;t++) {
        struct TwoMatrixOperationsRowArgs* thread_arg = malloc(sizeof(struct TwoMatrixOperationsRowArgs));
        thread_arg->M = M;
        thread_arg->h_start = t*H;
        if (thread_arg->h_start >= height) break;
        thread_arg->h_end = (t+1)*H-1;
        if (thread_arg->h_end >= height) thread_arg->h_end = height - 1;
        int create_error = pthread_create(&thread[t],&attr,init2DMatrixZeroRow,(void*) thread_arg);
        if (create_error) {
            printf("ERROR: Create thread failed.\n");
            exit(-1);
        }
    void* status;
    for(int n=0;n<t;n++) {
        int join_error = pthread_join(thread[n],&status);
        if (join_error) {
            printf("ERROR: Join thread failed.\n");
            exit(-1);
        }
    } 
    pthread_attr_destroy(&attr);
    M->initialized = true;
    return 0;
}

void* init2DMatrixZeroRow(void* args) {
    struct TwoMatrixOperationsRowArgs* a = (struct TwoMatrixOperationsRowArgs*) args;
    TwoDMatrix* M = a->M;
    for(int i=a->h_start;i<=a->h_end;i++) {
        M->d[i] = (float*) calloc(M->width,sizeof(float));
    }
    free(args);
    pthread_exit(NULL);
}

int init2DMatrixOne_MT(TwoDMatrix* M, int height, int width, int number_of_threads) {
    if (M->initialized) return 0;
    M->height = height;
    M->width = width;
    float** data = (float**) calloc(height, sizeof(float*));
    M->d = data;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_t* thread = malloc(sizeof(pthread_t)*number_of_threads);
    int H = height / number_of_threads + 1;
    int t = 0;
    for(;t<number_of_threads;t++) {
        struct TwoMatrixOperationsRowArgs* thread_arg = malloc(sizeof(struct TwoMatrixOperationsRowArgs));
        thread_arg->M = M;
        thread_arg->h_start = t*H;
        if (thread_arg->h_start >= height) break;
        thread_arg->h_end = (t+1)*H-1;
        if (thread_arg->h_end >= height) thread_arg->h_end = height - 1;
        int create_error = pthread_create(&thread[t],&attr,init2DMatrixOneRow,(void*) thread_arg);
        if (create_error) {
            printf("ERROR: Create thread failed.\n");
            exit(-1);
        }
    void* status;
    for(int n=0;n<t;n++) {
        int join_error = pthread_join(thread[n],&status);
        if (join_error) {
            printf("ERROR: Join thread failed.\n");
            exit(-1);
        }
    } 
    pthread_attr_destroy(&attr);
    M->initialized = true;
    return 0;
}

void* init2DMatrixOneRow(void* args) {
    struct TwoMatrixOperationsRowArgs* a = (struct TwoMatrixOperationsRowArgs*) args;
    TwoDMatrix* M = a->M;
    for(int i=a->h_start;i<=a->h_end;i++) {
        M->d[i] = (float*) calloc(M->width,sizeof(float));
        for(int j=0;j<M->width;j++) {
            M->d[i][j] = 1;
        }
    }
    free(args);
    pthread_exit(NULL);
}

int init2DMatrixNormRand_MT(TwoDMatrix* M, int height, int width, float mean, float std, int n, int number_of_threads) {
    if (M->initialized) return 0;
    M->height = height;
    M->width = width;
    float** data = (float**) calloc(height, sizeof(float*));
    M->d = data;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_t* thread = malloc(sizeof(pthread_t)*number_of_threads);
    int H = height / number_of_threads + 1;
    int t = 0;
    for(;t<number_of_threads;t++) {
        struct TwoMatrixOperationsRowArgs* thread_arg = malloc(sizeof(struct TwoMatrixOperationsRowArgs));
        thread_arg->M = M;
        thread_arg->mean = mean;
        thread_arg->std = std;
        thread_arg->n = n;
        thread_arg->h_start = t*H;
        if (thread_arg->h_start >= height) break;
        thread_arg->h_end = (t+1)*H-1;
        if (thread_arg->h_end >= height) thread_arg->h_end = height - 1;
        int create_error = pthread_create(&thread[t],&attr,init2DMatrixRow,(void*) thread_arg);
        if (create_error) {
            printf("ERROR: Create thread failed.\n");
            exit(-1);
        }
    void* status;
    for(int n=0;n<t;n++) {
        int join_error = pthread_join(thread[n],&status);
        if (join_error) {
            printf("ERROR: Join thread failed.\n");
            exit(-1);
        }
    } 
    pthread_attr_destroy(&attr);
    M->initialized = true;
    return 0;
}

void* init2DMatrixNormRandRow(void* args) {
    struct TwoMatrixOperationsRowArgs* a = (struct TwoMatrixOperationsRowArgs*) args;
    TwoDMatrix* M = a->M;
    for(int i=a->h_start;i<=a->h_end;i++) {
        M->d[i] = (float*) calloc(M->width,sizeof(float));
        for(int j=0;j<M->width;j++) {
            M->d[i][j] = random_normal(a->mean,a->std)*sqrt(2.0/a->n);
        }
    }
    free(args);
    pthread_exit(NULL);
}

int dotProduct_MT(TwoDMatrix* X, TwoDMatrix* M, TwoDMatrix* OUT, int number_of_threads) {
    if (X->width != M->height) {
        return 1;
    }
    init2DMatrix_MT(OUT,X->height,M->width,number_of_threads);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_t* thread = malloc(sizeof(pthread_t)*number_of_threads);
    int H = X->height / number_of_threads + 1;
    int t = 0;
    for(;t<number_of_threads;t++) {
        struct TwoMatrixOperationsRowArgs* thread_arg = malloc(sizeof(struct TwoMatrixOperationsRowArgs));
        thread_arg->X = X;
        thread_arg->M = M;
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
    pthread_attr_destroy(&attr);
    return 0;
}

void* dotProductRow(void* args) {
    struct TwoMatrixOperationsRowArgs* a = (struct TwoMatrixOperationsRowArgs*) args;
    TwoDMatrix* X = a->X;
    TwoDMatrix* M = a->M;
    TwoDMatrix* OUT = a->OUT;
    for(int i=a->h_start;i<=a->h_end;i++) {
        for(int j=0;j<M->width;j++) {
            float sum = 0;
            for(int p=0;p<X->width;p++) sum += X->d[i][p]*M->d[p][j];
            OUT->d[i][j] = sum;
        }
    }
    free(args);
    pthread_exit(NULL);
}

int transpose2DMatrix_MT(TwoDMatrix* M,TwoDMatrix* OUT, int number_of_threads) {
    init2DMatrix_MT(OUT, M->width,M->height,number_of_threads);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_t* thread = malloc(sizeof(pthread_t)*number_of_threads);
    int H = M->height / number_of_threads + 1;
    int t = 0;
    for(;t<number_of_threads;t++) {
        struct TwoMatrixOperationsRowArgs* thread_arg = malloc(sizeof(struct TwoMatrixOperationsRowArgs));
        thread_arg->M = M;
        thread_arg->OUT = OUT;
        thread_arg->h_start = t*H;
        if (thread_arg->h_start >= X->height) break;
        thread_arg->h_end = (t+1)*H-1;
        if (thread_arg->h_end >= X->height) thread_arg->h_end = X->height - 1;
        int create_error = pthread_create(&thread[t],&attr,transpose2DMatrixRow,(void*) thread_arg);
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
    pthread_attr_destroy(&attr);
    return 0;
}

void* transpose2DMatrixRow(void* args) {
    struct TwoMatrixOperationsRowArgs* a = (struct TwoMatrixOperationsRowArgs*) args;
    TwoDMatrix* M = a->M;
    TwoDMatrix* OUT = a->OUT;
    for(int i=a->h_start;i<=a->h_end;i++) {
        for(int j=0;j<M->width;j++) OUT->d[j][i] = M->d[i][j];
    }
    free(args);
    pthread_exit(NULL);
}

int twoDMatrixOperationMultithreadWrapperMOUT(TwoDMatrix* M, TwoDMatrix* OUT, void* (*func)(void *), int number_of_threads) {
    init2DMatrix_MT(OUT, M->width,M->height,number_of_threads);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_t* thread = malloc(sizeof(pthread_t)*number_of_threads);
    int H = M->height / number_of_threads + 1;
    int t = 0;
    for(;t<number_of_threads;t++) {
        struct TwoMatrixOperationsRowArgs* thread_arg = malloc(sizeof(struct TwoMatrixOperationsRowArgs));
        thread_arg->M = M;
        thread_arg->OUT = OUT;
        thread_arg->h_start = t*H;
        if (thread_arg->h_start >= X->height) break;
        thread_arg->h_end = (t+1)*H-1;
        if (thread_arg->h_end >= X->height) thread_arg->h_end = X->height - 1;
        int create_error = pthread_create(&thread[t],&attr,func,(void*) thread_arg);
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
    pthread_attr_destroy(&attr);
    return 0;
}