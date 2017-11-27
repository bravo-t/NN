#ifndef __MATRIX_OPT_MT_HEADER__
#define __MATRIX_OPT_MT_HEADER__

int init2DMatrix_MT(TwoDMatrix* M, int height, int width, int h_start, int h_end, bool* mem_allocated);
int init2DMatrixNormRand_MT(TwoDMatrix* M, int height, int width, float mean, float std, int n,int h_start, int h_end, bool* mem_allocated);
int init2DMatrixZero_MT(TwoDMatrix* M, int height, int width,int h_start, int h_end, bool* mem_allocated);
int init2DMatrixOne_MT(TwoDMatrix* M, int height, int width,int h_start, int h_end, bool* mem_allocated);
int copyTwoDMatrix_MT(TwoDMatrix* M, TwoDMatrix* OUT,int h_start, int h_end, bool* mem_allocated);
int transpose2DMatrix_MT(TwoDMatrix* M,TwoDMatrix* OUT,int h_start, int h_end, bool* mem_allocated);
int dotProduct_MT(TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* OUT,int h_start, int h_end, bool* mem_allocated);
int elementwiseAdd2DMatrix_MT(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT,int h_start, int h_end, bool* mem_allocated);
int elementwiseSub2DMatrix_MT(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT,int h_start, int h_end, bool* mem_allocated);
int elementwiseMul2DMatrix_MT(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT,int h_start, int h_end, bool* mem_allocated);
int elementwiseDiv2DMatrix_MT(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT,int h_start, int h_end, bool* mem_allocated);
int elementExp_MT(TwoDMatrix* M,TwoDMatrix* OUT,int h_start, int h_end, bool* mem_allocated);
int elementAdd_MT(TwoDMatrix* M, float a,TwoDMatrix* OUT,int h_start, int h_end, bool* mem_allocated);
int elementMul_MT(TwoDMatrix* M, float a,TwoDMatrix* OUT,int h_start, int h_end, bool* mem_allocated);
int elementDiv_MT(TwoDMatrix* M, float a,TwoDMatrix* OUT,int h_start, int h_end, bool* mem_allocated);
int broadcastAdd_MT(TwoDMatrix* M, TwoDMatrix* b, int direction, TwoDMatrix* OUT, int h_start, int h_end, bool* mem_allocated);
int broadcastSub_MT(TwoDMatrix* M, TwoDMatrix* b, int direction, TwoDMatrix* OUT, int h_start, int h_end, bool* mem_allocated);
int broadcastMul_MT(TwoDMatrix* M, TwoDMatrix* b, int direction, TwoDMatrix* OUT, int h_start, int h_end, bool* mem_allocated);
int broadcastDiv_MT(TwoDMatrix* M, TwoDMatrix* b, int direction, TwoDMatrix* OUT, int h_start, int h_end, bool* mem_allocated);

#endif
