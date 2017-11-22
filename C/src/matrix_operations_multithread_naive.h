#ifndef __OPERATION_HEADER_MT__
#define __OPERATION_HEADER_MT__

typedef struct {
    TwoDMatrix* X;
    TwoDMatrix* M;
    TwoDMatrix* OUT;
    float f;
    float mean;
    float std;
    int n;
    int h_start;
    int h_end;
} TwoDMatrixOperationsRowArgs;

typedef struct {
    ThreeDMatrix* X;
    ThreeDMatrix* M;
    ThreeDMatrix* OUT;
    float a;
    float mean;
    float std;
    int n;
    int h_start;
    int h_end;
} ThreeDMatrixOperationsRowArgs;

int init2DMatrix_MT(TwoDMatrix* M, int height, int width, int number_of_threads);
void* init2DMatrixRow(void* args);
int init2DMatrixZero_MT(TwoDMatrix* M, int height, int width, int number_of_threads);
void* init2DMatrixZeroRow(void* args);
int init2DMatrixOne_MT(TwoDMatrix* M, int height, int width, int number_of_threads);
void* init2DMatrixOneRow(void* args);
int init2DMatrixNormRand_MT(TwoDMatrix* M, int height, int width, float mean, float std, int n, int number_of_threads);
void* init2DMatrixNormRandRow(void* args);
int dotProduct_MT(TwoDMatrix* X, TwoDMatrix* M, TwoDMatrix* OUT, int number_of_threads);
void* dotProductRow(void* args);
int transpose2DMatrix_MT(TwoDMatrix* M,TwoDMatrix* OUT, int number_of_threads);
void* transpose2DMatrixRow(void* args);
int twoDMatrixOperationMultithreadWrapper(TwoDMatrixOperationsRowArgs* args, int height, int out_height, int out_width, void* (*func)(void *), int number_of_threads);
void* elementwiseAdd2DMatrixRow(void* args);
void* elementwiseSub2DMatrixRow(void* args);
void* elementwiseMul2DMatrixRow(void* args);
void* elementwiseDiv2DMatrixRow(void* args);
int elementwiseAdd2DMatrix_MT(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT, int number_of_threads);
int elementwiseSub2DMatrix_MT(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT, int number_of_threads);
int elementwiseMul2DMatrix_MT(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT, int number_of_threads);
int elementwiseDiv2DMatrix_MT(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT, int number_of_threads);
void* elementLeakyReLURow(void* args);
int elementLeakyReLU_MT(TwoDMatrix* M, float alpha, TwoDMatrix* OUT, int number_of_threads);
void* broadcastMatrixXRow(void* args);
void* broadcastMatrixYRow(void* args);
int broadcastMatrixX_MT(TwoDMatrix* M, int n, TwoDMatrix* OUT, int number_of_threads);
int broadcastMatrixY_MT(TwoDMatrix* M, int n, TwoDMatrix* OUT, int number_of_threads);
int broadcastMatrix_MT(TwoDMatrix* M, int n, int direction, TwoDMatrix* OUT, int number_of_threads);
void* elementExpRow(void* args);
int elementExp_MT(TwoDMatrix* M, TwoDMatrix* OUT, int number_of_threads);
void* elementAddRow(void* args);
void* elementSubRow(void* args);
void* elementMulRow(void* args);
void* elementDivRow(void* args);
int elementAdd_MT(TwoDMatrix* M, float a, TwoDMatrix* OUT, int number_of_threads);
int elementSub_MT(TwoDMatrix* M, float a, TwoDMatrix* OUT, int number_of_threads);
int elementMul_MT(TwoDMatrix* M, float a, TwoDMatrix* OUT, int number_of_threads);
int elementDiv_MT(TwoDMatrix* M, float a, TwoDMatrix* OUT, int number_of_threads);
void* elementSqrtRow(void* args);
int elementSqrt_MT(TwoDMatrix* M, TwoDMatrix* OUT, int number_of_threads);
int broadcastAdd_MT(TwoDMatrix* M, TwoDMatrix* b, int direction, TwoDMatrix* OUT, int number_of_threads);
int broadcastSub_MT(TwoDMatrix* M, TwoDMatrix* b, int direction, TwoDMatrix* OUT, int number_of_threads);
int broadcastMul_MT(TwoDMatrix* M, TwoDMatrix* b, int direction, TwoDMatrix* OUT, int number_of_threads);
int broadcastDiv_MT(TwoDMatrix* M, TwoDMatrix* b, int direction, TwoDMatrix* OUT, int number_of_threads);
void* destroy2DMatrixRow(void* args);
int destroy2DMatrix_MT(TwoDMatrix* M, int number_of_threads);
int init3DMatrix_MT(ThreeDMatrix* M, int depth, int height, int width, int number_of_threads);
void* init3DMatrixRow(void* args);
void* init3DMatrixZeroRow(void* args);
void* init3DMatrixOneRow(void* args);
void* init3DMatrixRandNormRow(void* args);
int init3DMatrixZero_MT(ThreeDMatrix* M, int depth, int height, int width, int number_of_threads);
int init3DMatrixOne_MT(ThreeDMatrix* M, int depth, int height, int width, int number_of_threads);
int init3DMatrixNormRand_MT(ThreeDMatrix* M, int depth, int height, int width, float mean, float std, int n, int number_of_threads);



#endif