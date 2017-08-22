#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "matrix_type.h"
#include "matrix_operations.h"
#include "convnet_operations.h"

int calcOutputSize(int length, int padding, int filter_length, int stride) {
    int tmp = length - filter_length + padding*2;
    if (tmp % stride != 0) {
        printf("ERROR: Illegal input size or stride settings: Input length = %d, filter length = %d, stride = %d\n",length, filter_length, stride);
        return 0;
    } else {
        return tmp / stride + 1;
    }
}

ThreeDMatrix* zeroPadding(ThreeDMatrix* X, int padding_height, int padding_width) {
    ThreeDMatrix* out = matrixMalloc(sizeof(ThreeDMatrix));
    init3DMatrix(out, X->depth, X->height + padding_height*2, X->width + padding_width*2);
    for(int i=0;i<out->depth;i++) {
        for(int j=0;j<out->height;j++) {
            for(int k=0;k<out->width;k++) {
                if ((j<padding_height || j>=padding_height+X->height) ||
                    (k<padding_width || k>=padding_width+X->width)) {
                    out->d[i][j][k] = 0;
                } else {
                    out->d[i][j][k] = X->d[i][j-padding_height][i-padding_width];
                }
            }
        }
    }
    destroy3DMatrix(X);
    return out;
}

int convSingleFilter(ThreeDMatrix* X,ThreeDMatrix* F,ThreeDMatrix* b, int stride_y, int stride_x,float** out) {
    int x_iter = ((X->width) - (F->width)) / stride_x;
    int y_iter = ((X->height) - (F->height)) / stride_y;
    for (int i=0; i<y_iter; i++) {
        for(int j=0;j<x_iter;j++) {
            float sum = 0;
            for(int l=0;l<F->depth;l++) {
                for(int m=0;m<F->height;m++) {
                    for(int n=0;n<F->width;n++) {
                        int x_m = i*stride_y+m;
                        int x_n = j*stride_x+n;
                        
                    }
                }
            }
        }
    }
}
