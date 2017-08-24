#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "matrix_type.h"
#include "matrix_operations.h"
#include "convnet_operations.h"
#include "convnet_layers.h"

ThreeDMatrix* convLayerForward(ThreeDMatrix* X, ThreeDMatrix** F, int number_of_filters, ThreeDMatrix** b, int f_height, int f_width, int stride_y, int stride_x, int padding_y, int padding_x) {
    ThreeDMatrix* V = matrixMalloc(sizeof(ThreeDMatrix));
    int V_height = calcOutputSize(X->height,padding_y,f_height,stride_y);
    int V_width = calcOutputSize(X->width,padding_x,f_width,stride_x);
    init3DMatrix(V, number_of_filters, V_height, V_width);
    ThreeDMatrix* X_padded = zeroPadding(X, padding_y, padding_x, false);
    for(int i=0;i<number_of_filters;i++) {
        convSingleFilter(X_padded,F[i],b[i],stride_y,stride_x,V->d[i]);
    }
    destroy3DMatrix(X_padded);
    return V;
}

ThreeDMatrix* maxPoolingForward(ThreeDMatrix* X, int stride_y, int stride_x, int pooling_width, int pooling_height) {
    ThreeDMatrix* V = matrixMalloc(sizeof(ThreeDMatrix));
    int V_height = calcOutputSize(X->height,0,pooling_height,stride_y); 
    int V_width = calcOutputSize(X->width,0,pooling_width,stride_x);
    init3DMatrix(V, X->depth, V_height, V_width);
    for(int i=0;i<V->depth;i++) {
        maxPoolingSingleSlice(X,pooling_height,pooling_width,stride_y,stride_x,i,V->d[i]);
    }
}
