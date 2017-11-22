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
        TwoDMatrixOperationsRowArgs* thread_arg = malloc(sizeof(TwoDMatrixOperationsRowArgs));
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
    TwoDMatrixOperationsRowArgs* a = (TwoDMatrixOperationsRowArgs*) args;
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
        TwoDMatrixOperationsRowArgs* thread_arg = malloc(sizeof(TwoDMatrixOperationsRowArgs));
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
    TwoDMatrixOperationsRowArgs* a = (TwoDMatrixOperationsRowArgs*) args;
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
        TwoDMatrixOperationsRowArgs* thread_arg = malloc(sizeof(TwoDMatrixOperationsRowArgs));
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
    TwoDMatrixOperationsRowArgs* a = (TwoDMatrixOperationsRowArgs*) args;
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
        TwoDMatrixOperationsRowArgs* thread_arg = malloc(sizeof(TwoDMatrixOperationsRowArgs));
        thread_arg->M = M;
        thread_arg->mean = mean;
        thread_arg->std = std;
        thread_arg->n = n;
        thread_arg->h_start = t*H;
        if (thread_arg->h_start >= height) break;
        thread_arg->h_end = (t+1)*H-1;
        if (thread_arg->h_end >= height) thread_arg->h_end = height - 1;
        int create_error = pthread_create(&thread[t],&attr,init2DMatrixNormRandRow,(void*) thread_arg);
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
    M->initialized = true;
    return 0;
}

void* init2DMatrixNormRandRow(void* args) {
    TwoDMatrixOperationsRowArgs* a = (TwoDMatrixOperationsRowArgs*) args;
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
        TwoDMatrixOperationsRowArgs* thread_arg = malloc(sizeof(TwoDMatrixOperationsRowArgs));
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
    TwoDMatrixOperationsRowArgs* a = (TwoDMatrixOperationsRowArgs*) args;
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
        TwoDMatrixOperationsRowArgs* thread_arg = malloc(sizeof(TwoDMatrixOperationsRowArgs));
        thread_arg->M = M;
        thread_arg->OUT = OUT;
        thread_arg->h_start = t*H;
        if (thread_arg->h_start >= M->height) break;
        thread_arg->h_end = (t+1)*H-1;
        if (thread_arg->h_end >= M->height) thread_arg->h_end = M->height - 1;
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
    TwoDMatrixOperationsRowArgs* a = (TwoDMatrixOperationsRowArgs*) args;
    TwoDMatrix* M = a->M;
    TwoDMatrix* OUT = a->OUT;
    for(int i=a->h_start;i<=a->h_end;i++) {
        for(int j=0;j<M->width;j++) OUT->d[j][i] = M->d[i][j];
    }
    free(args);
    pthread_exit(NULL);
}

int twoDMatrixOperationMultithreadWrapper(TwoDMatrixOperationsRowArgs* args, int height, int out_height, int out_width, void* (*func)(void *), int number_of_threads) {
    if(out_height != 0 && out_width != 0) init2DMatrix_MT(args->OUT, out_height,out_width,number_of_threads);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_t* thread = malloc(sizeof(pthread_t)*number_of_threads);
    int H = height / number_of_threads + 1;
    int t = 0;
    for(;t<number_of_threads;t++) {
        TwoDMatrixOperationsRowArgs* thread_arg = malloc(sizeof(TwoDMatrixOperationsRowArgs));
        memcpy(thread_arg, args, sizeof(TwoDMatrixOperationsRowArgs));
        thread_arg->h_start = t*H;
        if (thread_arg->h_start >= height) break;
        thread_arg->h_end = (t+1)*H-1;
        if (thread_arg->h_end >= height) thread_arg->h_end = height - 1;
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

void* elementwiseAdd2DMatrixRow(void* args) {
    TwoDMatrixOperationsRowArgs* a = (TwoDMatrixOperationsRowArgs*) args;
    TwoDMatrix* A = a->X;
    TwoDMatrix* B = a->M;
    TwoDMatrix* OUT = a->OUT;
    for(int i=a->h_start;i<=a->h_end;i++) {
        for(int j=0;j<A->width;j++) {
            OUT->d[i][j] = A->d[i][j] + B->d[i][j];
        }
    }
    free(args);
    pthread_exit(NULL);
}

void* elementwiseSub2DMatrixRow(void* args) {
    TwoDMatrixOperationsRowArgs* a = (TwoDMatrixOperationsRowArgs*) args;
    TwoDMatrix* A = a->X;
    TwoDMatrix* B = a->M;
    TwoDMatrix* OUT = a->OUT;
    for(int i=a->h_start;i<=a->h_end;i++) {
        for(int j=0;j<A->width;j++) {
            OUT->d[i][j] = A->d[i][j] - B->d[i][j];
        }
    }
    free(args);
    pthread_exit(NULL);
}

void* elementwiseMul2DMatrixRow(void* args) {
    TwoDMatrixOperationsRowArgs* a = (TwoDMatrixOperationsRowArgs*) args;
    TwoDMatrix* A = a->X;
    TwoDMatrix* B = a->M;
    TwoDMatrix* OUT = a->OUT;
    for(int i=a->h_start;i<=a->h_end;i++) {
        for(int j=0;j<A->width;j++) {
            OUT->d[i][j] = A->d[i][j] * B->d[i][j];
        }
    }
    free(args);
    pthread_exit(NULL);
}

void* elementwiseDiv2DMatrixRow(void* args) {
    TwoDMatrixOperationsRowArgs* a = (TwoDMatrixOperationsRowArgs*) args;
    TwoDMatrix* A = a->X;
    TwoDMatrix* B = a->M;
    TwoDMatrix* OUT = a->OUT;
    for(int i=a->h_start;i<=a->h_end;i++) {
        for(int j=0;j<A->width;j++) {
            OUT->d[i][j] = A->d[i][j] / B->d[i][j];
        }
    }
    free(args);
    pthread_exit(NULL);
}

int elementwiseAdd2DMatrix_MT(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT, int number_of_threads) {
    if (A->height != B->height) return 1;
    if (A->width != B->width) return 1;
    TwoDMatrixOperationsRowArgs* thread_arg = malloc(sizeof(TwoDMatrixOperationsRowArgs));
    thread_arg->X = A;
    thread_arg->M = B;
    thread_arg->OUT = OUT;
    twoDMatrixOperationMultithreadWrapper(thread_arg,
        A->height, 
        A->height, 
        A->width, 
        elementwiseAdd2DMatrixRow,
        number_of_threads);
    free(thread_arg);
    return 0;
}

int elementwiseSub2DMatrix_MT(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT, int number_of_threads) {
    if (A->height != B->height) return 1;
    if (A->width != B->width) return 1;
    TwoDMatrixOperationsRowArgs* thread_arg = malloc(sizeof(TwoDMatrixOperationsRowArgs));
    thread_arg->X = A;
    thread_arg->M = B;
    thread_arg->OUT = OUT;
    twoDMatrixOperationMultithreadWrapper(thread_arg,
        A->height, 
        A->height, 
        A->width, 
        elementwiseSub2DMatrixRow,
        number_of_threads);
    free(thread_arg);
    return 0;
}

int elementwiseMul2DMatrix_MT(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT, int number_of_threads) {
    if (A->height != B->height) return 1;
    if (A->width != B->width) return 1;
    TwoDMatrixOperationsRowArgs* thread_arg = malloc(sizeof(TwoDMatrixOperationsRowArgs));
    thread_arg->X = A;
    thread_arg->M = B;
    thread_arg->OUT = OUT;
    twoDMatrixOperationMultithreadWrapper(thread_arg,
        A->height, 
        A->height, 
        A->width, 
        elementwiseMul2DMatrixRow,
        number_of_threads);
    free(thread_arg);
    return 0;
}

int elementwiseDiv2DMatrix_MT(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT, int number_of_threads) {
    if (A->height != B->height) return 1;
    if (A->width != B->width) return 1;
    TwoDMatrixOperationsRowArgs* thread_arg = malloc(sizeof(TwoDMatrixOperationsRowArgs));
    thread_arg->X = A;
    thread_arg->M = B;
    thread_arg->OUT = OUT;
    twoDMatrixOperationMultithreadWrapper(thread_arg,
        A->height, 
        A->height, 
        A->width, 
        elementwiseDiv2DMatrixRow,
        number_of_threads);
    free(thread_arg);
    return 0;
}


void* elementLeakyReLURow(void* args) {
    TwoDMatrixOperationsRowArgs* a = (TwoDMatrixOperationsRowArgs*) args;
    TwoDMatrix* M = a->M;
    TwoDMatrix* OUT = a->OUT;
    float alpha = a->f;
    for(int i=a->h_start;i<=a->h_end;i++) {
        for(int j=0;j<M->width;j++) {
            if (M->d[i][j] >= 0) {
                OUT->d[i][j] = M->d[i][j];
            } else {
                OUT->d[i][j] = alpha * M->d[i][j];
            }
        }
    }
    free(args);
    pthread_exit(NULL);
}

int elementLeakyReLU_MT(TwoDMatrix* M, float alpha, TwoDMatrix* OUT, int number_of_threads) {
    TwoDMatrixOperationsRowArgs* thread_arg = malloc(sizeof(TwoDMatrixOperationsRowArgs));
    thread_arg->M = M;
    thread_arg->OUT = OUT;
    thread_arg->f = alpha;
    twoDMatrixOperationMultithreadWrapper(thread_arg,
        M->height, 
        M->height, 
        M->width, 
        elementLeakyReLURow,
        number_of_threads);
    free(thread_arg);
    return 0;
}

void* broadcastMatrixXRow(void* args) {
    TwoDMatrixOperationsRowArgs* a = (TwoDMatrixOperationsRowArgs*) args;
    TwoDMatrix* M = a->M;
    TwoDMatrix* OUT = a->OUT;
    int n = a->n;
    for(int i=a->h_start;i<=a->h_end;i++) {
        for(int j=0;j<n;j++) {
                OUT->d[i][j] = M->d[i][0];
        }
    }
    free(args);
    pthread_exit(NULL);
}

void* broadcastMatrixYRow(void* args) {
    TwoDMatrixOperationsRowArgs* a = (TwoDMatrixOperationsRowArgs*) args;
    TwoDMatrix* M = a->M;
    TwoDMatrix* OUT = a->OUT;
    for(int i=a->h_start;i<=a->h_end;i++) {
        for(int j=0;j<M->width;j++) {
                OUT->d[i][j] = M->d[0][j];
        }
    }
    free(args);
    pthread_exit(NULL);
}

int broadcastMatrixX_MT(TwoDMatrix* M, int n, TwoDMatrix* OUT, int number_of_threads) {
    TwoDMatrixOperationsRowArgs* thread_arg = malloc(sizeof(TwoDMatrixOperationsRowArgs));
    thread_arg->M = M;
    thread_arg->OUT = OUT;
    thread_arg->n = n;
    twoDMatrixOperationMultithreadWrapper(thread_arg,
        M->height, 
        M->height, 
        n, 
        broadcastMatrixXRow,
        number_of_threads);
    free(thread_arg);
    return 0;
}

int broadcastMatrixY_MT(TwoDMatrix* M, int n, TwoDMatrix* OUT, int number_of_threads) {
    TwoDMatrixOperationsRowArgs* thread_arg = malloc(sizeof(TwoDMatrixOperationsRowArgs));
    thread_arg->M = M;
    thread_arg->OUT = OUT;
    twoDMatrixOperationMultithreadWrapper(thread_arg,
        n, 
        n, 
        M->width, 
        broadcastMatrixYRow,
        number_of_threads);
    free(thread_arg);
    return 0;
}

int broadcastMatrix_MT(TwoDMatrix* M, int n, int direction, TwoDMatrix* OUT, int number_of_threads) {
    if (direction == 0) {
        if (M->width != 1) {
            printf("ERROR: Cannot horizontally broadcast matrix with a width that is not 1\n");
            return 1;
        }
        broadcastMatrixX_MT(M, n, OUT, number_of_threads);
    } else {
        if (M->height != 1) {
            printf("ERROR: Cannot vertically broadcast matrix with a height that is not 1\n");
            return 1;
        }
        broadcastMatrixY_MT(M, n, OUT, number_of_threads);
    }
    return 0;
}

void* elementExpRow(void* args) {
    TwoDMatrixOperationsRowArgs* a = (TwoDMatrixOperationsRowArgs*) args;
    TwoDMatrix* M = a->M;
    TwoDMatrix* OUT = a->OUT;
    for(int i=a->h_start;i<=a->h_end;i++) {
        for(int j=0;j<M->width;j++) {
                OUT->d[i][j] = exp(M->d[i][j]);
        }
    }
    free(args);
    pthread_exit(NULL);
}

int elementExp_MT(TwoDMatrix* M, TwoDMatrix* OUT, int number_of_threads) {
    TwoDMatrixOperationsRowArgs* thread_arg = malloc(sizeof(TwoDMatrixOperationsRowArgs));
    thread_arg->M = M;
    thread_arg->OUT = OUT;
    twoDMatrixOperationMultithreadWrapper(thread_arg,
        M->height, 
        M->height, 
        M->width, 
        elementExpRow,
        number_of_threads);
    free(thread_arg);
    return 0;
}

void* elementAddRow(void* args) {
    TwoDMatrixOperationsRowArgs* a = (TwoDMatrixOperationsRowArgs*) args;
    TwoDMatrix* M = a->M;
    TwoDMatrix* OUT = a->OUT;
    float z = a->f;
    for(int i=a->h_start;i<=a->h_end;i++) {
        for(int j=0;j<M->width;j++) {
                OUT->d[i][j] = M->d[i][j] + z;
        }
    }
    free(args);
    pthread_exit(NULL);
}

void* elementSubRow(void* args) {
    TwoDMatrixOperationsRowArgs* a = (TwoDMatrixOperationsRowArgs*) args;
    TwoDMatrix* M = a->M;
    TwoDMatrix* OUT = a->OUT;
    float z = a->f;
    for(int i=a->h_start;i<=a->h_end;i++) {
        for(int j=0;j<M->width;j++) {
                OUT->d[i][j] = M->d[i][j] - z;
        }
    }
    free(args);
    pthread_exit(NULL);
}

void* elementMulRow(void* args) {
    TwoDMatrixOperationsRowArgs* a = (TwoDMatrixOperationsRowArgs*) args;
    TwoDMatrix* M = a->M;
    TwoDMatrix* OUT = a->OUT;
    float z = a->f;
    for(int i=a->h_start;i<=a->h_end;i++) {
        for(int j=0;j<M->width;j++) {
                OUT->d[i][j] = M->d[i][j] * z;
        }
    }
    free(args);
    pthread_exit(NULL);
}

void* elementDivRow(void* args) {
    TwoDMatrixOperationsRowArgs* a = (TwoDMatrixOperationsRowArgs*) args;
    TwoDMatrix* M = a->M;
    TwoDMatrix* OUT = a->OUT;
    float z = a->f;
    for(int i=a->h_start;i<=a->h_end;i++) {
        for(int j=0;j<M->width;j++) {
                OUT->d[i][j] = M->d[i][j] / z;
        }
    }
    free(args);
    pthread_exit(NULL);
}

int elementAdd_MT(TwoDMatrix* M, float a, TwoDMatrix* OUT, int number_of_threads) {
    TwoDMatrixOperationsRowArgs* thread_arg = malloc(sizeof(TwoDMatrixOperationsRowArgs));
    thread_arg->M = M;
    thread_arg->OUT = OUT;
    thread_arg->f = a;
    twoDMatrixOperationMultithreadWrapper(thread_arg,
        M->height, 
        M->height, 
        M->width, 
        elementAddRow,
        number_of_threads);
    free(thread_arg);
    return 0;
}

int elementSub_MT(TwoDMatrix* M, float a, TwoDMatrix* OUT, int number_of_threads) {
    TwoDMatrixOperationsRowArgs* thread_arg = malloc(sizeof(TwoDMatrixOperationsRowArgs));
    thread_arg->M = M;
    thread_arg->OUT = OUT;
    thread_arg->f = a;
    twoDMatrixOperationMultithreadWrapper(thread_arg,
        M->height, 
        M->height, 
        M->width, 
        elementSubRow,
        number_of_threads);
    free(thread_arg);
    return 0;
}

int elementMul_MT(TwoDMatrix* M, float a, TwoDMatrix* OUT, int number_of_threads) {
    TwoDMatrixOperationsRowArgs* thread_arg = malloc(sizeof(TwoDMatrixOperationsRowArgs));
    thread_arg->M = M;
    thread_arg->OUT = OUT;
    thread_arg->f = a;
    twoDMatrixOperationMultithreadWrapper(thread_arg,
        M->height, 
        M->height, 
        M->width, 
        elementMulRow,
        number_of_threads);
    free(thread_arg);
    return 0;
}

int elementDiv_MT(TwoDMatrix* M, float a, TwoDMatrix* OUT, int number_of_threads) {
    TwoDMatrixOperationsRowArgs* thread_arg = malloc(sizeof(TwoDMatrixOperationsRowArgs));
    thread_arg->M = M;
    thread_arg->OUT = OUT;
    thread_arg->f = a;
    twoDMatrixOperationMultithreadWrapper(thread_arg,
        M->height, 
        M->height, 
        M->width, 
        elementDivRow,
        number_of_threads);
    free(thread_arg);
    return 0;
}

void* elementSqrtRow(void* args) {
    TwoDMatrixOperationsRowArgs* a = (TwoDMatrixOperationsRowArgs*) args;
    TwoDMatrix* M = a->M;
    TwoDMatrix* OUT = a->OUT;
    for(int i=a->h_start;i<=a->h_end;i++) {
        for(int j=0;j<M->width;j++) {
                OUT->d[i][j] = sqrt(M->d[i][j]);
        }
    }
    free(args);
    pthread_exit(NULL);
}

int elementSqrt_MT(TwoDMatrix* M, TwoDMatrix* OUT, int number_of_threads) {
    TwoDMatrixOperationsRowArgs* thread_arg = malloc(sizeof(TwoDMatrixOperationsRowArgs));
    thread_arg->M = M;
    thread_arg->OUT = OUT;
    twoDMatrixOperationMultithreadWrapper(thread_arg,
        M->height, 
        M->height, 
        M->width, 
        elementSqrtRow,
        number_of_threads);
    free(thread_arg);
    return 0;
}

int broadcastAdd_MT(TwoDMatrix* M, TwoDMatrix* b, int direction, TwoDMatrix* OUT, int number_of_threads) {
    TwoDMatrix *broadcasted = matrixMalloc(sizeof(TwoDMatrix));
    int n;
    if (direction == 0) {
        n = M->width;
    } else {
        n = M->height;
    }
    if (broadcastMatrix_MT(b,n,direction,broadcasted,number_of_threads)) {
        destroy2DMatrix_MT(broadcasted, number_of_threads);
        return 1;
    }
    if (elementwiseAdd2DMatrix_MT(M,broadcasted,OUT,number_of_threads)) {
        destroy2DMatrix_MT(broadcasted,number_of_threads);
        return 1;
    }
    destroy2DMatrix_MT(broadcasted,number_of_threads);
    return 0;
}

int broadcastSub_MT(TwoDMatrix* M, TwoDMatrix* b, int direction, TwoDMatrix* OUT, int number_of_threads) {
    TwoDMatrix *broadcasted = matrixMalloc(sizeof(TwoDMatrix));
    int n;
    if (direction == 0) {
        n = M->width;
    } else {
        n = M->height;
    }
    if (broadcastMatrix_MT(b,n,direction,broadcasted,number_of_threads)) {
        destroy2DMatrix_MT(broadcasted, number_of_threads);
        return 1;
    }
    if (elementwiseSub2DMatrix_MT(M,broadcasted,OUT,number_of_threads)) {
        destroy2DMatrix_MT(broadcasted,number_of_threads);
        return 1;
    }
    destroy2DMatrix_MT(broadcasted,number_of_threads);
    return 0;
}

int broadcastMul_MT(TwoDMatrix* M, TwoDMatrix* b, int direction, TwoDMatrix* OUT, int number_of_threads) {
    TwoDMatrix *broadcasted = matrixMalloc(sizeof(TwoDMatrix));
    int n;
    if (direction == 0) {
        n = M->width;
    } else {
        n = M->height;
    }
    if (broadcastMatrix_MT(b,n,direction,broadcasted,number_of_threads)) {
        destroy2DMatrix_MT(broadcasted, number_of_threads);
        return 1;
    }
    if (elementwiseMul2DMatrix_MT(M,broadcasted,OUT,number_of_threads)) {
        destroy2DMatrix_MT(broadcasted,number_of_threads);
        return 1;
    }
    destroy2DMatrix_MT(broadcasted,number_of_threads);
    return 0;
}

int broadcastDiv_MT(TwoDMatrix* M, TwoDMatrix* b, int direction, TwoDMatrix* OUT, int number_of_threads) {
    TwoDMatrix *broadcasted = matrixMalloc(sizeof(TwoDMatrix));
    int n;
    if (direction == 0) {
        n = M->width;
    } else {
        n = M->height;
    }
    if (broadcastMatrix_MT(b,n,direction,broadcasted,number_of_threads)) {
        destroy2DMatrix_MT(broadcasted, number_of_threads);
        return 1;
    }
    if (elementwiseDiv2DMatrix_MT(M,broadcasted,OUT,number_of_threads)) {
        destroy2DMatrix_MT(broadcasted,number_of_threads);
        return 1;
    }
    destroy2DMatrix_MT(broadcasted,number_of_threads);
    return 0;
}


void* destroy2DMatrixRow(void* args) {
    TwoDMatrixOperationsRowArgs* a = (TwoDMatrixOperationsRowArgs*) args;
    TwoDMatrix* M = a->M;
    for(int i=a->h_start;i<=a->h_end;i++) {
        free(M->d[i]);
        M->d[i] = NULL;
    }
    free(args);
    pthread_exit(NULL);
}

int destroy2DMatrix_MT(TwoDMatrix* M, int number_of_threads) {
    TwoDMatrixOperationsRowArgs* thread_arg = malloc(sizeof(TwoDMatrixOperationsRowArgs));
    thread_arg->M = M;
    twoDMatrixOperationMultithreadWrapper(thread_arg,
        M->height, 
        0, 
        0, 
        destroy2DMatrixRow,
        number_of_threads);
    free(M->d);
    M->d = NULL;
    free(M);
    M = NULL;
    free(thread_arg);
    return 0;
}

/*
int threeDMatrixOperationMultithreadWrapper(ThreeMatrixOperationsRowArgs* args, int height, int out_height, int out_width, void* (*func)(void *), int number_of_threads) {
    if(out_height != 0 && out_width != 0) init2DMatrix_MT(args->OUT, out_height,out_width,number_of_threads);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_t* thread = malloc(sizeof(pthread_t)*number_of_threads);
    int H = height / number_of_threads + 1;
    int t = 0;
    for(;t<number_of_threads;t++) {
        ThreeDMatrixOperationsRowArgs* thread_arg = malloc(sizeof(ThreeDMatrixOperationsRowArgs));
        memcpy(thread_arg, args, sizeof(ThreeDMatrixOperationsRowArgs));
        thread_arg->h_start = t*H;
        if (thread_arg->h_start >= height) break;
        thread_arg->h_end = (t+1)*H-1;
        if (thread_arg->h_end >= height) thread_arg->h_end = height - 1;
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
*/

int init3DMatrix_MT(ThreeDMatrix* M, int depth, int height, int width, int number_of_threads) {
    if (M->initialized) return 0;
    M->depth = depth;
    M->height = height;
    M->width = width;
    float*** data = calloc(depth, sizeof(float**));
    M->d = data;
    for(int i=0;i<=depth;i++) M->d[i] = (float**) calloc(height,sizeof(float*));
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_t* thread = malloc(sizeof(pthread_t)*number_of_threads);
    int H = height / number_of_threads + 1;
    int t = 0;
    for(;t<number_of_threads;t++) {
        ThreeDMatrixOperationsRowArgs* thread_arg = malloc(sizeof(ThreeDMatrixOperationsRowArgs));
        thread_arg->M = M;
        thread_arg->h_start = t*H;
        if (thread_arg->h_start >= height) break;
        thread_arg->h_end = (t+1)*H-1;
        if (thread_arg->h_end >= height) thread_arg->h_end = height - 1;
        int create_error = pthread_create(&thread[t],&attr,init3DMatrixRow,(void*) thread_arg);
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
    M->initialized = true;
    return 0;
}

void* init3DMatrixRow(void* args) {
    ThreeDMatrixOperationsRowArgs* a = (ThreeDMatrixOperationsRowArgs*) args;
    ThreeDMatrix* M = a->M;
    for(int i=0;i<M->depth;i++) {
        for(int j=a->h_start;j<a->h_end;j++) M->d[i][j] = (float*) calloc(M->width,sizeof(float));
    }
    free(args);
    pthread_exit(NULL);
}

void* init3DMatrixZeroRow(void* args) {
    ThreeDMatrixOperationsRowArgs* a = (ThreeDMatrixOperationsRowArgs*) args;
    ThreeDMatrix* M = a->M;
    for(int i=0;i<M->depth;i++) {
        for(int j=a->h_start;j<a->h_end;j++) M->d[i][j] = (float*) calloc(M->width,sizeof(float));
    }
    free(args);
    pthread_exit(NULL);
}

void* init3DMatrixOneRow(void* args) {
    ThreeDMatrixOperationsRowArgs* a = (ThreeDMatrixOperationsRowArgs*) args;
    ThreeDMatrix* M = a->M;
    for(int i=0;i<M->depth;i++) {
        for(int j=a->h_start;j<a->h_end;j++) {
            M->d[i][j] = (float*) calloc(M->width,sizeof(float));
            for(int k=0;k<M->width;k++) M->d[i][j][k] = 1;
        }
    }
    free(args);
    pthread_exit(NULL);
}

void* init3DMatrixRandNormRow(void* args) {
    ThreeDMatrixOperationsRowArgs* a = (ThreeDMatrixOperationsRowArgs*) args;
    ThreeDMatrix* M = a->M;
    for(int i=0;i<M->depth;i++) {
        for(int j=a->h_start;j<a->h_end;j++) {
            M->d[i][j] = (float*) calloc(M->width,sizeof(float));
            for(int k=0;k<M->width;k++) M->d[i][j][k] = random_normal(a->mean,a->std)*sqrt(2.0/a->n);
        }
    }
    free(args);
    pthread_exit(NULL);
}

int init3DMatrixZero_MT(ThreeDMatrix* M, int depth, int height, int width, int number_of_threads) {
    if (M->initialized) return 0;
    M->depth = depth;
    M->height = height;
    M->width = width;
    float*** data = calloc(depth, sizeof(float**));
    M->d = data;
    for(int i=0;i<=depth;i++) M->d[i] = (float**) calloc(height,sizeof(float*));
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_t* thread = malloc(sizeof(pthread_t)*number_of_threads);
    int H = height / number_of_threads + 1;
    int t = 0;
    for(;t<number_of_threads;t++) {
        ThreeDMatrixOperationsRowArgs* thread_arg = malloc(sizeof(ThreeDMatrixOperationsRowArgs));
        thread_arg->M = M;
        thread_arg->h_start = t*H;
        if (thread_arg->h_start >= height) break;
        thread_arg->h_end = (t+1)*H-1;
        if (thread_arg->h_end >= height) thread_arg->h_end = height - 1;
        int create_error = pthread_create(&thread[t],&attr,init3DMatrixZeroRow,(void*) thread_arg);
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
    M->initialized = true;
    return 0;
}

int init3DMatrixOne_MT(ThreeDMatrix* M, int depth, int height, int width, int number_of_threads) {
    if (M->initialized) return 0;
    M->depth = depth;
    M->height = height;
    M->width = width;
    float*** data = calloc(depth, sizeof(float**));
    M->d = data;
    for(int i=0;i<=depth;i++) M->d[i] = (float**) calloc(height,sizeof(float*));
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_t* thread = malloc(sizeof(pthread_t)*number_of_threads);
    int H = height / number_of_threads + 1;
    int t = 0;
    for(;t<number_of_threads;t++) {
        ThreeDMatrixOperationsRowArgs* thread_arg = malloc(sizeof(ThreeDMatrixOperationsRowArgs));
        thread_arg->M = M;
        thread_arg->h_start = t*H;
        if (thread_arg->h_start >= height) break;
        thread_arg->h_end = (t+1)*H-1;
        if (thread_arg->h_end >= height) thread_arg->h_end = height - 1;
        int create_error = pthread_create(&thread[t],&attr,init3DMatrixOneRow,(void*) thread_arg);
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
    M->initialized = true;
    return 0;
}

int init3DMatrixNormRand_MT(ThreeDMatrix* M, int depth, int height, int width, float mean, float std, int n, int number_of_threads) {
    if (M->initialized) return 0;
    M->depth = depth;
    M->height = height;
    M->width = width;
    float*** data = calloc(depth, sizeof(float**));
    M->d = data;
    for(int i=0;i<=depth;i++) M->d[i] = (float**) calloc(height,sizeof(float*));
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_t* thread = malloc(sizeof(pthread_t)*number_of_threads);
    int H = height / number_of_threads + 1;
    int t = 0;
    for(;t<number_of_threads;t++) {
        ThreeDMatrixOperationsRowArgs* thread_arg = malloc(sizeof(ThreeDMatrixOperationsRowArgs));
        thread_arg->M = M;
        thread_arg->mean = mean;
        thread_arg->std = std;
        thread_arg->n = n;
        thread_arg->h_start = t*H;
        if (thread_arg->h_start >= height) break;
        thread_arg->h_end = (t+1)*H-1;
        if (thread_arg->h_end >= height) thread_arg->h_end = height - 1;
        int create_error = pthread_create(&thread[t],&attr,init3DMatrixRandNormRow,(void*) thread_arg);
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
    M->initialized = true;
    return 0;
}
