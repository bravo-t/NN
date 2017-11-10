#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>
#include "src/network_type.h"
#include "src/matrix_operations.h"
#include "src/matrix_operations_multithread.h"

struct DotProductRowArgs {
    TwoDMatrix* X,
    TwoDMatrix* W,
    TwoDMatrix* OUT,
    int h
};
int dotProduct_MT(TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* OUT, int number_of_threads);

int main(int argc, char const *argv[])
{
	TwoDMatrix* X = matrixMalloc(sizeof(TwoDMatrix));
	TwoDMatrix* M = matrixMalloc(sizeof(TwoDMatrix));
	init2DMatrixNormRand(X,55,10000,0.0,1.0,2);
	init2DMatrixNormRand(M,10000,30,0.0,1.0,2);
	TwoDMatrix* OUT = matrixMalloc(sizeof(TwoDMatrix));
    dotProduct_MT(X,M,OUT,8);
	dotProduct_MT(X,M,OUT,64);
	return 0;
}

int dotProduct_MT(TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* OUT, int number_of_threads) {
    if (X->width != W->height) {
        return 1;
    }
    init2DMatrix(OUT,X->height,W->width);
    for(int i=0;i<X->height;i++) {
        for(;i%number_of_threads!=number_of_threads&&i<X->height;i++) {
            printf("DEBUG: thread = %d, i = %d\n", i%number_of_threads, i);
        }
    }

    return 0;
}

int* dotProductRow(void* args) {
    struct DotProductRowArgs* a = (struct DotProductRowArgs*) args;
    TwoDMatrix* X = a->X;
    TwoDMatrix* W = a->W;
    TwoDMatrix* OUT = a->OUT;
    int i = a->h;
    for(int j=0;j<W->width;j++) {
        float sum = 0;
        for(int p=0;p<X->width;p++) sum += X->d[i][p]*W->d[p][j];
        OUT->d[i][j] = sum;
    }
    free(args);
}