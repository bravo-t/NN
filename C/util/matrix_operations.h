#ifndef __OPERATION_HEADER__
#define __OPERATION_HEADER__
#include "matrix_type.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

float frand(void);
float random_normal(float mean, float std);
int init2DMatrix(TwoDMatrix* M, int height, int width);
int init2DMatrixNormRand(TwoDMatrix* M, int height, int width, float mean, float std);
int destroy2DMatrix(TwoDMatrix* M);
int transpose2DMatrix(TwoDMatrix* M,TwoDMatrix* OUT);
int dotProduct(TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* OUT);
int sumX2DMatrix(TwoDMatrix* M,TwoDMatrix* OUT);
int sumY2DMatrix(TwoDMatrix* M,TwoDMatrix* OUT);
int elementwiseAdd2DMatrix(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT);
int elementwiseMul2DMatrix(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT);
int elementExp(TwoDMatrix* M,TwoDMatrix* OUT);
int elementLeakyReLU(TwoDMatrix* M,float alpha, TwoDMatrix* OUT);

#endif
