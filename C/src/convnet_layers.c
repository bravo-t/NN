#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "network_type.h"
#include "matrix_operations.h"
#include "convnet_operations.h"
#include "convnet_layers.h"

/*
**           F,b
**            |
**            V
**      ------------
** X -> |   CONV   | -> V
**      ------------
*/

int convLayerForward(ThreeDMatrix* X, ThreeDMatrix** F, int number_of_filters, ThreeDMatrix** b, int f_height, int f_width, int stride_y, int stride_x, int padding_y, int padding_x, float alpha, ThreeDMatrix* V) {
    //ThreeDMatrix* V = matrixMalloc(sizeof(ThreeDMatrix));
    int V_height = calcOutputSize(X->height,padding_y,f_height,stride_y);
    int V_width = calcOutputSize(X->width,padding_x,f_width,stride_x);
    init3DMatrix(V, number_of_filters, V_height, V_width);
    ThreeDMatrix* X_padded = matrixMalloc(sizeof(ThreeDMatrix));
    zeroPadding(X, padding_y, padding_x,X_padded);
    for(int i=0;i<number_of_filters;i++) {
        convSingleFilter(X_padded,F[i],b[i],stride_y,stride_x,V->d[i]);
    }
    convReLUForward(V, alpha, V);
    destroy3DMatrix(X_padded);
    return 0;
}

/*
**             X    F,b   V
**             |     |    |
**             V     V    V
**             ------------
** dF,db,dX <- |   CONV   | <- dV
**             ------------
*/
int convLayerBackward(ThreeDMatrix* X, 
    ThreeDMatrix* V,
    ThreeDMatrix** F, 
    ThreeDMatrix* dV, 
    int padding_y, 
    int padding_x, 
    int stride_y, 
    int stride_x, 
    float alpha,
    ThreeDMatrix* dX, 
    ThreeDMatrix** dF, 
    ThreeDMatrix* db) {
    ThreeDMatrix* X_padded = matrixMalloc(sizeof(ThreeDMatrix));
    zeroPadding(X, padding_y, padding_x,X_padded);
    ThreeDMatrix* dX_padded = matrixMalloc(ThreeDMatrix);
    init3DMatrix(dX_padded, X->depth, X->height + 2*padding_y, X->width + 2*padding_x);
    init3DMatrix(dX, X->depth, X->height, X->width);
    for(int i=0;i<dV->depth;i++) {
        init3DMatrix(dF[i],F[i]->depth,F[i]->height, F[i]->width);
    }
    init3DMatrix(db,1,1,1);
    convReLUBackword(dV,V,alpha,dV);
    for(int z=0;z<dV->depth;z++) {
        convSingleFilterBackward(X_padded,F[z], dV,stride_y, stride_x, z, dX_padded, dF[z], db);
    }
    unpad(dX_padded, padding_y, padding_x, dX);
    destroy3DMatrix(X_padded);
    destroy3DMatrix(dX_padded);
    return 0;
}
/*      -----------
** X -> | POOLING | -> V
**      -----------
*/
int maxPoolingForward(ThreeDMatrix* X, int stride_y, int stride_x, int pooling_width, int pooling_height, ThreeDMatrix* V) {
    //ThreeDMatrix* V = matrixMalloc(sizeof(ThreeDMatrix));
    int V_height = calcOutputSize(X->height,0,pooling_height,stride_y); 
    int V_width = calcOutputSize(X->width,0,pooling_width,stride_x);
    init3DMatrix(V, X->depth, V_height, V_width);
    for(int i=0;i<V->depth;i++) {
        maxPoolingSingleSlice(X,pooling_height,pooling_width,stride_y,stride_x,i,V->d[i]);
    }
    return 0;
}

/*
**            X
**            |
**            V
**       -----------
** dX <- | POOLING | <- dV
**       -----------
*/
int maxPoolingBackword(ThreeDMatrix* dV, ThreeDMatrix* X, int stride_y, int stride_x, int pooling_width, int pooling_height, ThreeDMatrix* dX) {
    init3DMatrix(dX, X->depth, X->height, X->width);
    for(int z=0;z<X->depth;z++) {
        maxPoolingSingleSliceBackword(X, dV, pooling_height, pooling_width, stride_y, stride_x, z, dX->d[z]);
    }
    return 0;
}