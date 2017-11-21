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
	TwoDMatrix* X = matrixMalloc(sizeof(TwoDMatrix));
	TwoDMatrix* M = matrixMalloc(sizeof(TwoDMatrix));
	init2DMatrixNormRand(X,10,5,0.0,1.0,2);
	init2DMatrixNormRand(M,5,3,0.0,1.0,2);
    TwoDMatrix* col_vec =  matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* row_vec = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrixNormRand(col_vec,10,1,0.0,1.0,2);
    init2DMatrixNormRand(row_vec,1,5,0.0,1.0,2);

    printf("Testing dotProduct\n");
	TwoDMatrix* OUT_MT = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* OUT_ST = matrixMalloc(sizeof(TwoDMatrix)); 

    dotProduct_MT(X,M,OUT_MT,4);
    dotProduct(X,M,OUT_ST);
	//dotProduct_MT(X,M,OUT,64);
    checkMatrixDiff(OUT_MT,OUT_ST,1e-3);
    destroy2DMatrix_MT(OUT_MT,4);
    destroy2DMatrix_MT(OUT_ST,4);
    
    printf("Testing transpose\n");
    OUT_MT = matrixMalloc(sizeof(TwoDMatrix));
    OUT_ST = matrixMalloc(sizeof(TwoDMatrix));
    transpose2DMatrix_MT(X,OUT_MT,4);
    transpose2DMatrix(X,OUT_ST);
    checkMatrixDiff(OUT_MT,OUT_ST,1e-3);
    destroy2DMatrix_MT(OUT_MT,4);
    destroy2DMatrix_MT(OUT_ST,4);
    

    printf("Testing elementwiseAdd\n");
    OUT_MT = matrixMalloc(sizeof(TwoDMatrix));
    OUT_ST = matrixMalloc(sizeof(TwoDMatrix));
    elementwiseAdd2DMatrix_MT(X,M,OUT_MT,4);
    elementwiseAdd2DMatrix(X,M,OUT_ST);
    checkMatrixDiff(OUT_MT,OUT_ST,1e-3);
    destroy2DMatrix_MT(OUT_MT,4);
    destroy2DMatrix_MT(OUT_ST,4);

    printf("Testing elementwiseSub\n");
    OUT_MT = matrixMalloc(sizeof(TwoDMatrix));
    OUT_ST = matrixMalloc(sizeof(TwoDMatrix));
    elementwiseSub2DMatrix_MT(X,M,OUT_MT,4);
    elementwiseSub2DMatrix(X,M,OUT_ST);
    checkMatrixDiff(OUT_MT,OUT_ST,1e-3);
    destroy2DMatrix_MT(OUT_MT,4);
    destroy2DMatrix_MT(OUT_ST,4);

    printf("Testing elementwiseMul\n");
    OUT_MT = matrixMalloc(sizeof(TwoDMatrix));
    OUT_ST = matrixMalloc(sizeof(TwoDMatrix));
    elementwiseMul2DMatrix_MT(X,M,OUT_MT,4);
    elementwiseMul2DMatrix(X,M,OUT_ST);
    checkMatrixDiff(OUT_MT,OUT_ST,1e-3);
    destroy2DMatrix_MT(OUT_MT,4);
    destroy2DMatrix_MT(OUT_ST,4);

    printf("Testing elementwiseDiv\n");
    OUT_MT = matrixMalloc(sizeof(TwoDMatrix));
    OUT_ST = matrixMalloc(sizeof(TwoDMatrix));
    elementwiseDiv2DMatrix_MT(X,M,OUT_MT,4);
    elementwiseDiv2DMatrix(X,M,OUT_ST);
    checkMatrixDiff(OUT_MT,OUT_ST,1e-3);
    destroy2DMatrix_MT(OUT_MT,4);
    destroy2DMatrix_MT(OUT_ST,4);
    
    printf("Testing elementLeakyReLU\n");
    OUT_MT = matrixMalloc(sizeof(TwoDMatrix));
    OUT_ST = matrixMalloc(sizeof(TwoDMatrix));
    elementLeakyReLU_MT(X,0.01,OUT_MT,4);
    elementLeakyReLU(X,0.01,OUT_ST);
    checkMatrixDiff(OUT_MT,OUT_ST,1e-3);
    destroy2DMatrix_MT(OUT_MT,4);
    destroy2DMatrix_MT(OUT_ST,4);
    
    printf("Testing broadcastMatrix in X dir\n");
    OUT_MT = matrixMalloc(sizeof(TwoDMatrix));
    OUT_ST = matrixMalloc(sizeof(TwoDMatrix));
    broadcastMatrix_MT(col_vec,10,0,OUT_MT,4);
    broadcastMatrix(col_vec,10,0,OUT_ST);
    checkMatrixDiff(OUT_MT,OUT_ST,1e-3);
    destroy2DMatrix_MT(OUT_MT,4);
    destroy2DMatrix_MT(OUT_ST,4);

    printf("Testing broadcastMatrix in Y dir\n");
    OUT_MT = matrixMalloc(sizeof(TwoDMatrix));
    OUT_ST = matrixMalloc(sizeof(TwoDMatrix));
    broadcastMatrix_MT(row_vec,10,1,OUT_MT,4);
    broadcastMatrix(row_vec,10,1,OUT_ST);
    checkMatrixDiff(OUT_MT,OUT_ST,1e-3);
    destroy2DMatrix_MT(OUT_MT,4);
    destroy2DMatrix_MT(OUT_ST,4);

    printf("Testing elementExp\n");
    OUT_MT = matrixMalloc(sizeof(TwoDMatrix));
    OUT_ST = matrixMalloc(sizeof(TwoDMatrix));
    elementExp_MT(X,OUT_MT,4);
    elementExp(X,OUT_ST);
    checkMatrixDiff(OUT_MT,OUT_ST,1e-3);
    destroy2DMatrix_MT(OUT_MT,4);
    destroy2DMatrix_MT(OUT_ST,4);

    printf("Testing elementSqrt\n");
    OUT_MT = matrixMalloc(sizeof(TwoDMatrix));
    OUT_ST = matrixMalloc(sizeof(TwoDMatrix));
    elementSqrt_MT(X,OUT_MT,4);
    elementSqrt(X,OUT_ST);
    checkMatrixDiff(OUT_MT,OUT_ST,1e-3);
    destroy2DMatrix_MT(OUT_MT,4);
    destroy2DMatrix_MT(OUT_ST,4);
    
    printf("Testing elementAdd\n");
    OUT_MT = matrixMalloc(sizeof(TwoDMatrix));
    OUT_ST = matrixMalloc(sizeof(TwoDMatrix));
    elementAdd_MT(X,0.123,OUT_MT,4);
    elementAdd(X,0.123,OUT_ST);
    checkMatrixDiff(OUT_MT,OUT_ST,1e-3);
    destroy2DMatrix_MT(OUT_MT,4);
    destroy2DMatrix_MT(OUT_ST,4);

/*
    printf("Testing elementSub\n");
    OUT_MT = matrixMalloc(sizeof(TwoDMatrix));
    OUT_ST = matrixMalloc(sizeof(TwoDMatrix));
    elementSub_MT(X,0.123,OUT_MT,4);
    elementSub(X,0.123,OUT_ST);
    checkMatrixDiff(OUT_MT,OUT_ST,1e-3);
    destroy2DMatrix_MT(OUT_MT,4);
    destroy2DMatrix_MT(OUT_ST,4);
*/
    printf("Testing elementMul\n");
    OUT_MT = matrixMalloc(sizeof(TwoDMatrix));
    OUT_ST = matrixMalloc(sizeof(TwoDMatrix));
    elementMul_MT(X,0.123,OUT_MT,4);
    elementMul(X,0.123,OUT_ST);
    checkMatrixDiff(OUT_MT,OUT_ST,1e-3);
    destroy2DMatrix_MT(OUT_MT,4);
    destroy2DMatrix_MT(OUT_ST,4);

    printf("Testing elementDiv\n");
    OUT_MT = matrixMalloc(sizeof(TwoDMatrix));
    OUT_ST = matrixMalloc(sizeof(TwoDMatrix));
    elementDiv_MT(X,0.123,OUT_MT,4);
    elementDiv(X,0.123,OUT_ST);
    checkMatrixDiff(OUT_MT,OUT_ST,1e-3);
    destroy2DMatrix_MT(OUT_MT,4);
    destroy2DMatrix_MT(OUT_ST,4);
    
    printf("Testing broadcastAdd in X dir\n");
    OUT_MT = matrixMalloc(sizeof(TwoDMatrix));
    OUT_ST = matrixMalloc(sizeof(TwoDMatrix));
    broadcastAdd_MT(X,col_vec,0,OUT_MT,4);
    broadcastAdd(X,col_vec,0,OUT_ST);
    checkMatrixDiff(OUT_MT,OUT_ST,1e-3);
    destroy2DMatrix_MT(OUT_MT,4);
    destroy2DMatrix_MT(OUT_ST,4);

    printf("Testing broadcastAdd in Y dir\n");
    OUT_MT = matrixMalloc(sizeof(TwoDMatrix));
    OUT_ST = matrixMalloc(sizeof(TwoDMatrix));
    broadcastAdd_MT(X,row_vec,1,OUT_MT,4);
    broadcastAdd(X,row_vec,1,OUT_ST);
    checkMatrixDiff(OUT_MT,OUT_ST,1e-3);
    destroy2DMatrix_MT(OUT_MT,4);
    destroy2DMatrix_MT(OUT_ST,4);

    printf("Testing broadcastSub in X dir\n");
    OUT_MT = matrixMalloc(sizeof(TwoDMatrix));
    OUT_ST = matrixMalloc(sizeof(TwoDMatrix));
    broadcastSub_MT(X,col_vec,0,OUT_MT,4);
    broadcastSub(X,col_vec,0,OUT_ST);
    checkMatrixDiff(OUT_MT,OUT_ST,1e-3);
    destroy2DMatrix_MT(OUT_MT,4);
    destroy2DMatrix_MT(OUT_ST,4);

    printf("Testing broadcastSub in Y dir\n");
    OUT_MT = matrixMalloc(sizeof(TwoDMatrix));
    OUT_ST = matrixMalloc(sizeof(TwoDMatrix));
    broadcastSub_MT(X,row_vec,1,OUT_MT,4);
    broadcastSub(X,row_vec,1,OUT_ST);
    checkMatrixDiff(OUT_MT,OUT_ST,1e-3);
    destroy2DMatrix_MT(OUT_MT,4);
    destroy2DMatrix_MT(OUT_ST,4);

    printf("Testing broadcastMul in X dir\n");
    OUT_MT = matrixMalloc(sizeof(TwoDMatrix));
    OUT_ST = matrixMalloc(sizeof(TwoDMatrix));
    broadcastMul_MT(X,col_vec,0,OUT_MT,4);
    broadcastMul(X,col_vec,0,OUT_ST);
    checkMatrixDiff(OUT_MT,OUT_ST,1e-3);
    destroy2DMatrix_MT(OUT_MT,4);
    destroy2DMatrix_MT(OUT_ST,4);

    printf("Testing broadcastMul in Y dir\n");
    OUT_MT = matrixMalloc(sizeof(TwoDMatrix));
    OUT_ST = matrixMalloc(sizeof(TwoDMatrix));
    broadcastMul_MT(X,row_vec,1,OUT_MT,4);
    broadcastMul(X,row_vec,1,OUT_ST);
    checkMatrixDiff(OUT_MT,OUT_ST,1e-3);
    destroy2DMatrix_MT(OUT_MT,4);
    destroy2DMatrix_MT(OUT_ST,4);

    printf("Testing broadcastDiv in X dir\n");
    OUT_MT = matrixMalloc(sizeof(TwoDMatrix));
    OUT_ST = matrixMalloc(sizeof(TwoDMatrix));
    broadcastDiv_MT(X,col_vec,0,OUT_MT,4);
    broadcastDiv(X,col_vec,0,OUT_ST);
    checkMatrixDiff(OUT_MT,OUT_ST,1e-3);
    destroy2DMatrix_MT(OUT_MT,4);
    destroy2DMatrix_MT(OUT_ST,4);

    printf("Testing broadcastDiv in Y dir\n");
    OUT_MT = matrixMalloc(sizeof(TwoDMatrix));
    OUT_ST = matrixMalloc(sizeof(TwoDMatrix));
    broadcastDiv_MT(X,row_vec,1,OUT_MT,4);
    broadcastDiv(X,row_vec,1,OUT_ST);
    checkMatrixDiff(OUT_MT,OUT_ST,1e-3);
    destroy2DMatrix_MT(OUT_MT,4);
    destroy2DMatrix_MT(OUT_ST,4);
    
	return 0;
}


