#ifndef __MATRIX_OPT_MT_HEADER__
#define __MATRIX_OPT_MT_HEADER__

int calc_h_start(int id, int height,int number_of_threads);
int calc_h_end(int id, int height,int number_of_threads);
void reset_mem_allocated(int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
void preset_mem_allocated(int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
void* matrixMalloc_thread(char* share_memory_name, int size, int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
void* malloc_thread(char* share_memory_name, int size, int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
void* calloc_thread(char* share_memory_name, int n, int blk_size, int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
int init2DMatrix_thread(TwoDMatrix* M, int height, int width, int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
int init2DMatrixNormRand_thread(TwoDMatrix* M, int height, int width, float mean, float std, int n,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
int init2DMatrixZero_thread(TwoDMatrix* M, int height, int width,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
int init2DMatrixOne_thread(TwoDMatrix* M, int height, int width,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
int destroy2DMatrix_thread(TwoDMatrix* M, int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
int copyTwoDMatrix_thread(TwoDMatrix* M, TwoDMatrix* OUT, int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
int transpose2DMatrix_thread(TwoDMatrix* M,TwoDMatrix* OUT,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
int dotProduct_thread(TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* OUT,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
int elementwiseAdd2DMatrix_thread(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
int elementwiseSub2DMatrix_thread(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
int elementwiseMul2DMatrix_thread(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
int elementwiseDiv2DMatrix_thread(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
int elementSqrt_thread(TwoDMatrix* M,TwoDMatrix* OUT,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
int elementExp_thread(TwoDMatrix* M,TwoDMatrix* OUT,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
int elementLeakyReLU_thread(TwoDMatrix* M, float alpha,TwoDMatrix* OUT,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
int elementAdd_thread(TwoDMatrix* M, float a,TwoDMatrix* OUT,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
int elementMul_thread(TwoDMatrix* M, float a,TwoDMatrix* OUT,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
int elementDiv_thread(TwoDMatrix* M, float a,TwoDMatrix* OUT,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
int broadcastAdd_thread(TwoDMatrix* M, TwoDMatrix* b, int direction, TwoDMatrix* OUT, int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
int broadcastSub_thread(TwoDMatrix* M, TwoDMatrix* b, int direction, TwoDMatrix* OUT, int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
int broadcastMul_thread(TwoDMatrix* M, TwoDMatrix* b, int direction, TwoDMatrix* OUT, int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
int broadcastDiv_thread(TwoDMatrix* M, TwoDMatrix* b, int direction, TwoDMatrix* OUT, int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
int sumX2DMatrix_thread(TwoDMatrix* M,TwoDMatrix* OUT,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
int maxX2DMatrix_thread(TwoDMatrix* M,TwoDMatrix* OUT,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
int sumY2DMatrix_thread(TwoDMatrix* M,TwoDMatrix* OUT,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
int maxY2DMatrix_thread(TwoDMatrix* M,TwoDMatrix* OUT,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
float sumAll_thread(TwoDMatrix* M,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);

int init3DMatrix_thread(ThreeDMatrix* M, int depth, int height, int width, int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
int init3DMatrixNormRand_thread(ThreeDMatrix* M, int depth, int height, int width,float mean, float std, int n, int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
int init3DMatrixZero_thread(ThreeDMatrix* M, int depth,int height, int width,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
int init3DMatrixOne_thread(ThreeDMatrix* M, int depth, int height, int width,float mean, float std, int n, int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);
int destroy3DMatrix_thread(ThreeDMatrix* M, int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier);


#endif
