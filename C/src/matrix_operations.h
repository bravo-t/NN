#ifndef __OPERATION_HEADER__
#define __OPERATION_HEADER__
#include "matrix_type.h"

#ifdef DEBUG
#define debugPrintMatrix(M) __debugPrintMatrix(M,#M)
#else
#define debugPrintMatrix(M) ((void)0)
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

TwoDMatrix* matrixMalloc(int size);
float frand(void);
float random_normal(float mean, float std);
int init2DMatrix(TwoDMatrix* M, int height, int width);
int init2DMatrixNormRand(TwoDMatrix* M, int height, int width, float mean, float std);
int init2DMatrixZero(TwoDMatrix* M, int height, int width);
int init2DMatrixOne(TwoDMatrix* M, int height, int width);
int copyTwoDMatrix(TwoDMatrix* M, TwoDMatrix* OUT);
int destroy2DMatrix(TwoDMatrix* M);
int transpose2DMatrix(TwoDMatrix* M,TwoDMatrix* OUT);
int dotProduct(TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* OUT);
int sumX2DMatrix(TwoDMatrix* M,TwoDMatrix* OUT);
int maxX2DMatrix(TwoDMatrix* M,TwoDMatrix* OUT);
int sumY2DMatrix(TwoDMatrix* M,TwoDMatrix* OUT);
int maxY2DMatrix(TwoDMatrix* M,TwoDMatrix* OUT);
float sumAll(TwoDMatrix* M);
int elementwiseAdd2DMatrix(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT);
int elementwiseSub2DMatrix(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT);
int elementwiseMul2DMatrix(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT);
int elementwiseDiv2DMatrix(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT);
int elementExp(TwoDMatrix* M,TwoDMatrix* OUT);
int elementAdd(TwoDMatrix* M, float a,TwoDMatrix* OUT);
int elementMul(TwoDMatrix* M, float a,TwoDMatrix* OUT);
int elementDiv(TwoDMatrix* M,float a, TwoDMatrix* OUT);
int elementSqrt(TwoDMatrix* M, TwoDMatrix* OUT);
int elementLeakyReLU(TwoDMatrix* M,float alpha, TwoDMatrix* OUT);
int broadcastMatrix(TwoDMatrix* M, int n, int direction, TwoDMatrix* OUT);
int broadcastAdd(TwoDMatrix* M, TwoDMatrix* b, int direction, TwoDMatrix* OUT);
int broadcastSub(TwoDMatrix* M, TwoDMatrix* b, int direction, TwoDMatrix* OUT);
int broadcastMul(TwoDMatrix* M, TwoDMatrix* b, int direction, TwoDMatrix* OUT);
int broadcastDiv(TwoDMatrix* M, TwoDMatrix* b, int direction, TwoDMatrix* OUT);
int chop2DMatrix(TwoDMatrix* M, int height_start, int height_end, TwoDMatrix* OUT);
int matrixYMeanVar(TwoDMatrix* M, TwoDMatrix* mean, TwoDMatrix* var);
int matrixXMeanVar(TwoDMatrix* M, TwoDMatrix* mean, TwoDMatrix* var);
int init3DMatrix(ThreeDMatrix* M, int depth, int height, int width);
int init3DMatrixNormRand(ThreeDMatrix* M, int depth, int height, int width, float mean, float std);
int destroy3DMatrix (ThreeDMatrix* M);
int threeDMatrixElementMul(ThreeDMatrix* X, float n, ThreeDMatrix* OUT);
int threeDMatrixElementwiseAdd(ThreeDMatrix* A, ThreeDMatrix* B, ThreeDMatrix* OUT);
ThreeDMatrix* chop3DMatrix(ThreeDMatrix* X, int start_y, int start_x, int end_y, int end_x);
int assign3DMatrix(ThreeDMatrix* in, int start_y, int start_x, int end_y, int end_x, ThreeDMatrix* X);

#endif
