#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "matrix_type.h"
#include "matrix_operations.h"
#include "convnet_operations.h"
#include "convnet_layers.h"

ThreeDMatrix* convLayerForward(ThreeDMatrix* X, ThreeDMatrix** F, int number_of_filters, ThreeDMatrix** b, int f_height, int f_width, int stride_y, int stride_x) {
    ThreeDMatrix* V = matrixMalloc(sizeof(ThreeDMatrix));
    int V_height = calcOutputSize(X->height,0,f_height,stride_y);
    int V_width = calcOutputSize(X->width,0,f_width,stride_x);
    init3DMatrix(V, number_of_filters, V_height, V_width);
    for(int i=0;i<number_of_filters;i++) {
        convSingleFilter(X,F[i],b[i],stride,V->d[i]);
    }
    return V;
}
