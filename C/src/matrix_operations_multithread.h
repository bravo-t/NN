#ifndef __MATRIX_OPT_MT_HEADER__
#define __MATRIX_OPT_MT_HEADER__

int calc_h_start(int id, int height);
int calc_h_end(int id, int height);
void reset_mem_allocated(int id, bool* mem_allocated);
void preset_mem_allocated(int id, bool* mem_allocated);
void* matrixMalloc_thread(char* share_memory_name, int size, int id);
int init2DMatrix_thread(TwoDMatrix* M, int height, int width, int h_start, int h_end, bool* mem_allocated);
int init2DMatrixNormRand_thread(TwoDMatrix* M, int height, int width, float mean, float std, int n,int h_start, int h_end, bool* mem_allocated);
int init2DMatrixZero_thread(TwoDMatrix* M, int height, int width,int h_start, int h_end, bool* mem_allocated);
int init2DMatrixOne_thread(TwoDMatrix* M, int height, int width,int h_start, int h_end, bool* mem_allocated);
int destroy2DMatrix_thread(TwoDMatrix* M, int h_start, int h_end, bool* mem_allocated);
int copyTwoDMatrix_thread(TwoDMatrix* M, TwoDMatrix* OUT, int id, bool* mem_allocated);
int transpose2DMatrix_thread(TwoDMatrix* M,TwoDMatrix* OUT,int id, bool* mem_allocated);
int dotProduct_thread(TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* OUT,int id, bool* mem_allocated);
int elementwiseAdd2DMatrix_thread(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT,int id, bool* mem_allocated);
int elementwiseSub2DMatrix_thread(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT,int id, bool* mem_allocated);
int elementwiseMul2DMatrix_thread(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT,int id, bool* mem_allocated);
int elementwiseDiv2DMatrix_thread(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT,int id, bool* mem_allocated);
int elementExp_thread(TwoDMatrix* M,TwoDMatrix* OUT,int id, bool* mem_allocated);
int elementLeakyReLU_thread(TwoDMatrix* M, float alpha,TwoDMatrix* OUT,int id, bool* mem_allocated);
int elementAdd_thread(TwoDMatrix* M, float a,TwoDMatrix* OUT,int id, bool* mem_allocated);
int elementMul_thread(TwoDMatrix* M, float a,TwoDMatrix* OUT,int id, bool* mem_allocated);
int elementDiv_thread(TwoDMatrix* M, float a,TwoDMatrix* OUT,int id, bool* mem_allocated);
int broadcastAdd_thread(TwoDMatrix* M, TwoDMatrix* b, int direction, TwoDMatrix* OUT, int id, bool* mem_allocated);
int broadcastSub_thread(TwoDMatrix* M, TwoDMatrix* b, int direction, TwoDMatrix* OUT, int id, bool* mem_allocated);
int broadcastMul_thread(TwoDMatrix* M, TwoDMatrix* b, int direction, TwoDMatrix* OUT, int id, bool* mem_allocated);
int broadcastDiv_thread(TwoDMatrix* M, TwoDMatrix* b, int direction, TwoDMatrix* OUT, int id, bool* mem_allocated);
int sumX2DMatrix_thread(TwoDMatrix* M,TwoDMatrix* OUT,int id, bool* mem_allocated);
int maxX2DMatrix_thread(TwoDMatrix* M,TwoDMatrix* OUT,int id, bool* mem_allocated);
int sumY2DMatrix_thread(TwoDMatrix* M,TwoDMatrix* OUT,int id, bool* mem_allocated);
int maxY2DMatrix_thread(TwoDMatrix* M,TwoDMatrix* OUT,int id, bool* mem_allocated);

#endif
