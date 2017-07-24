#include <stdio.h>
#include "util/misc_utils.h"
#include "util/matrix_operations.h"

void printMatrix(TwoDMatrix *M) {
    printf("Height of matrix: %d, width: %d\n",M->height,M->width);
    for(int i=0;i<M->height;i++) {
        for(int j=0;j<M->width;j++) {
            printf("%f\t",M->d[i][j]);
        }
        printf("\n");
    }
}

int main() {
    TwoDMatrix *test1 = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrixNormRand(test1,3,5,0.0,1.0);
    TwoDMatrix *test2 = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrixNormRand(test2,5,3,0.0,1.0);
    printMatrix(test1);
    printMatrix(test2);
    TwoDMatrix *dot = matrixMalloc(sizeof(TwoDMatrix));
    dotProduct(test1,test2,dot);
    printMatrix(dot);
    TwoDMatrix *addX = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix *addY = matrixMalloc(sizeof(TwoDMatrix));
    sumX2DMatrix(test1,addX);
    sumY2DMatrix(test1,addY);
    printMatrix(addX);
    printMatrix(addY);
    destroy2DMatrix(test1);
    destroy2DMatrix(test2);
    destroy2DMatrix(dot);
    return 0;
}
