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

int dotProduct_MT(TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* OUT, int number_of_threads) {
    if (X->width != W->height) {
        return 1;
    }
    init2DMatrix(OUT,X->height,W->width);
    for(int i=0;i<X->height;i++) {
    	for(int n=i%number_of_threads;n!=number_of_threads&&i<X->height;i++) {
    		printf("DEBUG: thread = %d, i = %d\n", n, i);
    	}
    }

    return 0;
}

int main(int argc, char const *argv[])
{
	TwoDMatrix* X = matrixMalloc(sizeof(TwoDMatrix));
	TwoDMatrix* M = matrixMalloc(sizeof(TwoDMatrix));
	init2DMatrixNormRand(X,55,10000,0.0,1.0,2);
	init2DMatrixNormRand(M,10000,30,0.0,1.0,2);
	TwoDMatrix* OUT = matrixMalloc(sizeof(TwoDMatrix));
	dotProduct_MT(X,M,OUT,8);
	return 0;
}