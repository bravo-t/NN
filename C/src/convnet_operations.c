#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
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
    //if (destroy_original) destroy3DMatrix(X);
    return out;
}

int unpad(ThreeDMatrix* padded, int padding_height, int padding_width, ThreeDMatrix* out) {
    //ThreeDMatrix* out = matrixMalloc(sizeof(ThreeDMatrix));
    init3DMatrix(out, padded->depth, padded->height - padding_height*2, padded->width - padding_width*2);
    for(int i=0;i<out->depth;i++) {
        for(int j=0;j<out->height;j++) {
            for(int k=0;k<out->width;k++) {
                out->d[i][j][k] = padded->d[i][j+padding_height][k+padding_width];
            }
        }
    }
    return 0;
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
                        sum += F->d[l][m][n] * X->d[l][x_m][x_n];
                    }
                }
            }
            sum += b->d[0][0][0];
            out[i][j] = sum;
        }
    }
    return 0;
}

int maxPoolingSingleSlice(ThreeDMatrix* X, int pooling_height, int pooling_width, int stride_y, int stride_x,int z, float** out) {
    int x_iter = ((X->width) - pooling_width) / stride_x;
    int y_iter = ((X->height) - pooling_height) / stride_y;
    for (int i=0; i<y_iter; i++) {
        for(int j=0;j<x_iter;j++) {
            float max = X->d[z][i*stride_y][j*stride_x];
            for(int m=0;m<F->height;m++) {
                for(int n=0;n<F->width;n++) {
                    int x_m = i*stride_y+m;
                    int x_n = j*stride_x+n;
                    max = fmaxf(max, X->d[z][x_m][x_n])
                }
            }
            out[i][j] = sum;
        }
    }
    return 0;
}


int convSingleFilterBackward(ThreeDMatrix* X,
    ThreeDMatrix* F, 
    ThreeDMatrix* dV, 
    int stride_y, 
    int stride_x, 
    int z, 
    ThreeDMatrix* dX, 
    ThreeDMatrix* dF, 
    ThreeDMatrix* db) {
    int iter_y = ((X->height) - (F->height)) / stride_y;
    int iter_x = ((X->width) - (F->width)) / stride_x;
    for(int i=0;i<iter_y;i++) {
        for(int j=0;j<iter_x;j++) {
            db->d[0][0][0] += dV->d[z][i][j];
            int window_start_y = i * stride_y;
            int window_end_y = (i + 1) * stride_y - 1;
            int window_start_x = j * stride_x;
            int window_end_x = (j + 1) * stride_x - 1;
            for(int y=window_start_y;y<=window_end_y;y++) {
                for(int x=window_start_x;x<=window_end_x;x++) {
                    for(int depth=0;depth<X->depth;depth++) {
                        dF->[depth][y-window_start_y][x-window_start_x] += X->d[depth][y][x] * dV->d[z][i][j];
                        dX->[depth][y][x] += F->d[depth][y-window_start_y][x-window_start_x] * dV->d[z][i][j];
                    }
                }
            }
        }
    }
    return 0;
}

int threeDMatrix2Col(ThreeDMatrix* X, TwoDMatrix* OUT) {
    init2DMatrix(OUT, X->depth*X->height*X->width, 1);
    int count = 0;
    for(int i=0;i<X->depth;i++) {
        for(int j=0;j<X->height;j++) {
            for(int k=0;k<X->width;k++) {
                OUT->d[count][0] = X->d[i][j][k];
                count++;
            }
        }
    }
    return 0;
}