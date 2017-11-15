#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include "network_type.h"
#include "matrix_operations.h"
#include "matrix_operations_multithread.h"
#include "layers_multithread.h"

int affineLayerForward_MT(TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* b, TwoDMatrix* OUT, int number_of_threads) {
    init2DMatrix_MT(OUT, X->height, W->width, number_of_threads);
    if (dotProduct_MT(X,W,OUT, number_of_threads)) {
        printf("ERROR: Input matrix size mismatch: X->width = %d, W->height = %d\n", X->width,W->height);
        exit(1);
    }
    broadcastAdd_MT(OUT, b, 1, OUT, number_of_threads);
    return 0;
}

int affineLayerBackword_MT(TwoDMatrix* dOUT, TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* b, TwoDMatrix* dX, TwoDMatrix* dW, TwoDMatrix* db, int number_of_threads) {
    init2DMatrix_MT(dX, X->height, X->width, number_of_threads);
    init2DMatrix_MT(dW, W->height, W->width, number_of_threads);
    init2DMatrix_MT(db, b->height, b->width, number_of_threads);
    TwoDMatrix* XT = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* WT = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(XT, X->width, X->height, number_of_threads);
    init2DMatrix_MT(WT, W->width, W->height, number_of_threads);
    transpose2DMatrix_MT(X, XT, number_of_threads);
    transpose2DMatrix_MT(W, WT, number_of_threads);
    if (dotProduct_MT(dOUT,WT,dX, number_of_threads)) {
        printf("ERROR: Input matrix size mismatch: dOUT->width = %d, W.T->height = %d\n", dOUT->width,WT->height);
        exit(1);
    }
    if (dotProduct_MT(XT,dOUT,dW, number_of_threads)) {
        printf("ERROR: Input matrix size mismatch: X.T->width = %d, dOUT->height = %d\n", XT->width,dOUT->height);
        exit(1);
    }
    if (db->height == 1) {
        sumY2DMatrix(dOUT,db);
    } else {
        sumX2DMatrix(dOUT,db);
    }
    destroy2DMatrix_MT(XT, number_of_threads);
    destroy2DMatrix_MT(WT, number_of_threads);
    return 0;
}

int leakyReLUForward_MT(TwoDMatrix* M, float alpha, TwoDMatrix* OUT, int number_of_threads) {
    return elementLeakyReLU_MT(M, alpha, OUT, number_of_threads);
}

int vanillaUpdate_MT(TwoDMatrix* M, TwoDMatrix* dM, float learning_rate, TwoDMatrix* OUT, int number_of_threads) {
    init2DMatrix_MT(OUT, M->height, M->width,number_of_threads);
    TwoDMatrix* dM_scaled = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(dM_scaled, M->height, M->width,number_of_threads);
    elementMul_MT(dM,learning_rate,dM_scaled,number_of_threads);
    int retval = elementwiseSub2DMatrix_MT(M, dM_scaled, OUT,number_of_threads);
    destroy2DMatrix_MT(dM_scaled,number_of_threads);
    return 0;
}

void* leakyReLUBackwardRow(void* args) {
    TwoDMatrixOperationsRowArgs* a = (TwoDMatrixOperationsRowArgs*) args;
    TwoDMatrix* dM = a->M;
    TwoDMatrix* M = a->X;
    TwoDMatrix* OUT = a->OUT;
    float alpha = a->f;
    for(int i=a->h_start;i<=a->h_end;i++) {
        for(int j=0;j<dM->width;j++) {
            if (M->d[i][j] >= 0) {
                OUT->d[i][j] = dM->d[i][j];
            } else {
                OUT->d[i][j] = alpha * dM->d[i][j];
            }
        }
    }
    free(args);
    pthread_exit(NULL);
}

int leakyReLUBackward_MT(TwoDMatrix* dM, TwoDMatrix* M, float alpha, TwoDMatrix* OUT, int number_of_threads) {
    TwoDMatrixOperationsRowArgs* thread_arg = malloc(sizeof(TwoDMatrixOperationsRowArgs));
    thread_arg->M = dM;
    thread_arg->X = M;
    thread_arg->OUT = OUT;
    thread_arg->f = alpha;
    twoDMatrixOperationMultithreadWrapper(thread_arg,
        M->height, 
        M->height, 
        M->width, 
        leakyReLUBackwardRow,
        number_of_threads);
    free(thread_arg);
    return 0;
}



