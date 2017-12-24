#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include "src/network_type.h"
#include "src/matrix_operations.h"
#include "src/matrix_operations_multithread.h"
#include "src/misc_utils.h"

int main(int argc, char const *argv[]) {
    srand(time(NULL));
    int number_of_threads = 4;
    int matrix_sizes = 100;
    int network_depth = 2;
    TwoDMatrix** Ws = malloc(sizeof(TwoDMatrix*)*network_depth);
    TwoDMatrix** bs = malloc(sizeof(TwoDMatrix*)*network_depth);
    TwoDMatrix** Hs = malloc(sizeof(TwoDMatrix*)*network_depth);
    TwoDMatrix** dWs = malloc(sizeof(TwoDMatrix*)*network_depth);
    TwoDMatrix** dbs = malloc(sizeof(TwoDMatrix*)*network_depth);
    TwoDMatrix** dHs = malloc(sizeof(TwoDMatrix*)*network_depth);
    for(int i=0;i<network_depth;i++) {
        Ws[i] = matrixMalloc(sizeof(TwoDMatrix));
        bs[i] = matrixMalloc(sizeof(TwoDMatrix));
        Hs[i] = matrixMalloc(sizeof(TwoDMatrix));
        dWs[i] = matrixMalloc(sizeof(TwoDMatrix));
        dbs[i] = matrixMalloc(sizeof(TwoDMatrix));
        dHs[i] = matrixMalloc(sizeof(TwoDMatrix));
        init2DMatrixNormRand(Ws[i],matrix_sizes,matrix_sizes,0.0,1.0,matrix_sizes);
        init2DMatrixNormRand(bs[i],1,matrix_sizes,0.0,1.0,matrix_sizes);
        init2DMatrix(Hs[i],matrix_sizes,matrix_sizes);
    }
    TwoDMatrix* X = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* M = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrixNormRand(X,matrix_sizes,matrix_sizes,0.0,1.0,2);

    thread_barrier_t inst_rdy = THREAD_BARRIER_INITIALIZER;
    thread_barrier_t inst_ack = THREAD_BARRIER_INITIALIZER;
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    ThreadControl* control_handle = initControlHandle(&mutex, &inst_rdy, &inst_ack, number_of_threads);

    

    
    return 0;
}


