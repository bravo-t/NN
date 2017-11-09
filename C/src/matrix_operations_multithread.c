#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>
#include "network_type.h"
#include "matrix_operations.h"

int dotProduct_MT(TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* OUT, int number_of_threads) {
    if (X->width != W->height) {
        return 1;
    }
    init2DMatrix(OUT,X->height,W->width);
    for(int i=0;i<X->height;i++) {
    	for(;i%number_of_threads!=number_of_threads&&i<X->height;i++) {

    	}
    }
    for(int i=0;i<X->height;i++) {
        for(int j=0;j<W->width;j++) {
            float sum = 0;
            for(int p=0;p<X->width;p++) sum += X->d[i][p]*W->d[p][j];
            OUT->d[i][j] = sum;
        }
    }
    return 0;
}