#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>
#include "thread_barrier.h"
#include "inter-process_communication.h"
#include "network_type.h"
#include "matrix_operations.h"
#include "matrix_operations_multithread.h"

extern pthread_mutex_t mutex;
extern pthread_cond_t cond;
extern pthread_barrier_t barrier;
extern int number_of_threads;

int calc_h_start(int id, int height,int number_of_threads) {
    return(id*height/number_of_threads);
}

int calc_h_end(int id, int height,int number_of_threads) {
    int h_end = ((id+1)*height/number_of_threads)-1;
    if (h_end < height) {
        return h_end;
    } else if (id == 0 && height < number_of_threads) {
        return height - 1;
    } else {
        // Return a value that makes no sense to prevent functions from over-writing
        return -1;
    }
}


/********************************************************************************************/
// reset_mem_allocated and preset_mem_allocated plays a very important role in this program //
// they reset and preset the global mem_allocated variable to prevent race conditions and   //
// spurious wakeups, as well as initialize the thread barrier to synchronize all threads    //
/********************************************************************************************/
void reset_mem_allocated(int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    if (id == 0) {
        pthread_mutex_lock(mutex);
        thread_barrier_init(barrier,number_of_threads);
        *mem_allocated = false;
        pthread_mutex_unlock(mutex);
    } else {
        pthread_mutex_lock(mutex);
        while((*mem_allocated)) pthread_cond_wait(cond,mutex);
        pthread_mutex_unlock(mutex);
    }
    if(id == 0) {
        pthread_mutex_lock(mutex);
        pthread_cond_broadcast(cond);
        pthread_mutex_unlock(mutex);
    }
    thread_barrier_wait_reinit(barrier,number_of_threads);
}

void preset_mem_allocated(int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    if (id == 0) {
        pthread_mutex_lock(mutex);
        thread_barrier_init(barrier,number_of_threads);
        *mem_allocated = true;
        pthread_mutex_unlock(mutex);
    } else {
        pthread_mutex_lock(mutex);
        while(!(*mem_allocated)) pthread_cond_wait(cond,mutex);
        pthread_mutex_unlock(mutex);
    }
    if(id == 0) {
        pthread_mutex_lock(mutex);
        pthread_cond_broadcast(cond);
        pthread_mutex_unlock(mutex);
    }
    thread_barrier_wait_reinit(barrier,number_of_threads);
}

void* matrixMalloc_thread(char* share_memory_name, int size, int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    reset_mem_allocated(id,mem_allocated,number_of_threads,mutex,cond,barrier);
    void* M;
    if (id == 0) {
        pthread_mutex_lock(mutex);
        thread_barrier_init(barrier,number_of_threads);
        void* memory_chunk = matrixMalloc(size);
        printf("DEBUG: %s: id %d: Writing shm: %p\n",share_memory_name,id,memory_chunk);
        IPCWriteToSharedMem(share_memory_name,&memory_chunk,sizeof(void*));
        *mem_allocated = true;
        printf("DEBUG: %s: id %d: Shm created\n",share_memory_name,id);
        pthread_mutex_unlock(mutex);
    } else {
        pthread_mutex_lock(mutex);
        printf("DEBUG: %s: id %d: Wait for shm\n",share_memory_name,id);
        while(!(*mem_allocated)) pthread_cond_wait(cond,mutex);
        pthread_mutex_unlock(mutex);
    }
    if(id == 0) {
        pthread_mutex_lock(mutex);
        printf("DEBUG: %s: id %d: Signal other threads\n",share_memory_name,id);
        pthread_cond_broadcast(cond);
        pthread_mutex_unlock(mutex);
    }
    pthread_mutex_lock(mutex);
    printf("DEBUG: %s: id %d: Read from shm\n",share_memory_name,id);
    IPCReadFromSharedMem(share_memory_name,&M,sizeof(void*));
    printf("DEBUG: %s: id %d: Read content %p from shm\n",share_memory_name,id,M);
    pthread_mutex_unlock(mutex);
    thread_barrier_wait_reinit(barrier,number_of_threads);
    if (id == 0) {
        printf("DEBUG: %s: id %d: Removing shm\n",share_memory_name,id);
        IPCRemoveSharedMemFile(share_memory_name);
    }
    thread_barrier_wait_reinit(barrier,number_of_threads);
    return M;
}

void* malloc_thread(char* share_memory_name, int size, int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    reset_mem_allocated(id,mem_allocated,number_of_threads,mutex,cond,barrier);
    void* M;
    if (id == 0) {
        pthread_mutex_lock(mutex);
        thread_barrier_init(barrier,number_of_threads);
        void* memory_chunk = malloc(size);
        printf("DEBUG: id %d: Writing shm: %p\n",id,memory_chunk);
        IPCWriteToSharedMem(share_memory_name,&memory_chunk,sizeof(void*));
        *mem_allocated = true;
        printf("DEBUG: id %d: Shm created\n",id);
        pthread_mutex_unlock(mutex);
    } else {
        pthread_mutex_lock(mutex);
        printf("DEBUG: id %d: Wait for shm\n",id);
        while(!(*mem_allocated)) pthread_cond_wait(cond,mutex);

        pthread_mutex_unlock(mutex);
    }
    if(id == 0) {
        pthread_mutex_lock(mutex);
        printf("DEBUG: id %d: Signal other threads\n",id);
        pthread_cond_broadcast(cond);
        pthread_mutex_unlock(mutex);
    }
    pthread_mutex_lock(mutex);
    printf("DEBUG: id %d: Read from shm\n",id);
    IPCReadFromSharedMem(share_memory_name,&M,sizeof(void*));
    printf("DEBUG: id %d: Read content %p from shm\n",id,M);
    pthread_mutex_unlock(mutex);
    thread_barrier_wait_reinit(barrier,number_of_threads);
    if (id == 0) {
        printf("DEBUG: id %d: Removing shm\n",id);
        IPCRemoveSharedMemFile(share_memory_name);
    }
    thread_barrier_wait_reinit(barrier,number_of_threads);
    return M;
}

void* calloc_thread(char* share_memory_name, int n, int blk_size, int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    reset_mem_allocated(id,mem_allocated,number_of_threads,mutex,cond,barrier);
    void* M;
    if (id == 0) {
        pthread_mutex_lock(mutex);
        thread_barrier_init(barrier,number_of_threads);
        void* memory_chunk = calloc(n,blk_size);
        IPCWriteToSharedMem(share_memory_name,&memory_chunk,sizeof(void*));
        *mem_allocated = true;
        pthread_mutex_unlock(mutex);
    } else {
        pthread_mutex_lock(mutex);
        while(!(*mem_allocated)) pthread_cond_wait(cond,mutex);
        pthread_mutex_unlock(mutex);
    }
    if(id == 0) {
        pthread_mutex_lock(mutex);
        pthread_cond_broadcast(cond);
        pthread_mutex_unlock(mutex);
    }
    pthread_mutex_lock(mutex);
    IPCReadFromSharedMem(share_memory_name,&M,sizeof(void*));
    pthread_mutex_unlock(mutex);
    thread_barrier_wait_reinit(barrier,number_of_threads);
    if (id == 0) {
        IPCRemoveSharedMemFile(share_memory_name);
    }
    thread_barrier_wait_reinit(barrier,number_of_threads);
    return M;
}

// Change this to thread 0 only
int init2DMatrix_thread(TwoDMatrix* M, int height, int width, int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    reset_mem_allocated(id,mem_allocated,number_of_threads,mutex,cond,barrier);
    if (M->initialized) return 0;
    int h_start = calc_h_start(id,height,number_of_threads);
    int h_end = calc_h_end(id,height,number_of_threads);
    if (h_start == 0) {
        pthread_mutex_lock(mutex);
        printf("DEBUG: Thread %d will init (%d-%d)X(%d) of %dX%d\n",id,h_start,h_end,width,height,width);
        thread_barrier_init(barrier,number_of_threads);
        M->height = height;
        M->width = width;
        M->d = (float**) calloc(height, sizeof(float*));
        *mem_allocated = true;
        pthread_mutex_unlock(mutex);
    } else {
        pthread_mutex_lock(mutex);
        printf("DEBUG: Thread %d will init (%d-%d)X(%d) of %dX%d\n",id,h_start,h_end,width,height,width);
        while(!(*mem_allocated)) pthread_cond_wait(cond,mutex);
        pthread_mutex_unlock(mutex);
    }
    if(h_start == 0) {
        pthread_mutex_lock(mutex);
        pthread_cond_broadcast(cond);
        pthread_mutex_unlock(mutex);
    }
    // TODO
    // for(int i=h_start;i<=h_end && i<M->height;i++)
    for(int i=h_start;i<=h_end;i++) {
        M->d[i] = (float*) calloc(width,sizeof(float));
    }
    if(h_start == 0) M->initialized = true;
    thread_barrier_wait_reinit(barrier,number_of_threads);
    return 0;
}

int init2DMatrixNormRand_thread(TwoDMatrix* M, int height, int width, float mean, float std, int n,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    reset_mem_allocated(id,mem_allocated,number_of_threads,mutex,cond,barrier);
    if (M->initialized) return 0;
    int h_start = calc_h_start(id,height,number_of_threads);
    int h_end = calc_h_end(id,height,number_of_threads);
    if (h_start == 0) {
        pthread_mutex_lock(mutex);
        thread_barrier_init(barrier,number_of_threads);
        M->height = height;
        M->width = width;
        M->d = (float**) calloc(height, sizeof(float*));
        *mem_allocated = true;
        pthread_mutex_unlock(mutex);
    } else {
        pthread_mutex_lock(mutex);
        while(!(*mem_allocated)) pthread_cond_wait(cond,mutex);
        pthread_mutex_unlock(mutex);
    }
    if(h_start == 0) {
        pthread_mutex_lock(mutex);
        pthread_cond_broadcast(cond);
        pthread_mutex_unlock(mutex);
    }
    for(int i=h_start;i<=h_end;i++) {
        M->d[i] = (float*) calloc(width,sizeof(float));
        for(int j=0;j<width;j++) {
            M->d[i][j] = random_normal(mean,std)*sqrt(2.0/n);
        }
    }
    if(h_start == 0) M->initialized = true;
    thread_barrier_wait_reinit(barrier,number_of_threads);
    return 0;
}

int init2DMatrixZero_thread(TwoDMatrix* M, int height, int width,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    init2DMatrix_thread(M,height,width,id,mem_allocated,number_of_threads,mutex,cond,barrier);
    return 0;
}

int init2DMatrixOne_thread(TwoDMatrix* M, int height, int width,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    reset_mem_allocated(id,mem_allocated,number_of_threads,mutex,cond,barrier);
    if (M->initialized) return 0;
    int h_start = calc_h_start(id,height,number_of_threads);
    int h_end = calc_h_end(id,height,number_of_threads);
    if (h_start == 0) {
        pthread_mutex_lock(mutex);
        thread_barrier_init(barrier,number_of_threads);
        M->height = height;
        M->width = width;
        M->d = (float**) calloc(height, sizeof(float*));
        *mem_allocated = true;
        pthread_mutex_unlock(mutex);
    } else {
        pthread_mutex_lock(mutex);
        while(!(*mem_allocated)) pthread_cond_wait(cond,mutex);
        pthread_mutex_unlock(mutex);
    }
    if(h_start == 0) {
        pthread_mutex_lock(mutex);
        pthread_cond_broadcast(cond);
        pthread_mutex_unlock(mutex);
    }
    for(int i=h_start;i<=h_end;i++) {
        M->d[i] = (float*) calloc(width,sizeof(float));
        for(int j=0;j<width;j++) {
            M->d[i][j] = 1;
        }
    }
    if(h_start == 0) M->initialized = true;
    thread_barrier_wait_reinit(barrier,number_of_threads);
    return 0;
}

int destroy2DMatrix_thread(TwoDMatrix* M, int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    int h_start = calc_h_start(id,M->height,number_of_threads);
    int h_end = calc_h_end(id,M->height,number_of_threads);
    for(int i=h_start;i<=h_end;i++) {
        free(M->d[i]);
        M->d[i] = NULL;
    }
    preset_mem_allocated(id,mem_allocated,number_of_threads,mutex,cond,barrier);
    if (h_start == 0) {
        pthread_mutex_lock(mutex);
        thread_barrier_init(barrier,number_of_threads);
        free(M->d);
        M->d = NULL;
        free(M);
        M = NULL;
        *mem_allocated = false;
        pthread_mutex_unlock(mutex);
    } else {
        pthread_mutex_lock(mutex);
        while((*mem_allocated)) pthread_cond_wait(cond,mutex);
        pthread_mutex_unlock(mutex);
    }
    if(h_start == 0) {
        pthread_mutex_lock(mutex);
        pthread_cond_broadcast(cond);
        pthread_mutex_unlock(mutex);
    }
    thread_barrier_wait_reinit(barrier,number_of_threads);
    return 0;
}

int copyTwoDMatrix_thread(TwoDMatrix* M, TwoDMatrix* OUT, int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    int h_start = calc_h_start(id,M->height,number_of_threads);
    int h_end = calc_h_end(id,M->height,number_of_threads);
    reset_mem_allocated(id,mem_allocated,number_of_threads,mutex,cond,barrier);
    int retval = init2DMatrix_thread(OUT, M->height, M->width,id,mem_allocated,number_of_threads,mutex,cond,barrier);
    for(int i=h_start;i<=h_end;i++) {
        for(int j=0;j<M->width;j++) {
            OUT->d[i][j] = M->d[i][j];
        }
    }
    return retval;
}

int transpose2DMatrix_thread(TwoDMatrix* M,TwoDMatrix* OUT,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    int w_start = calc_h_start(id,M->width,number_of_threads);
    int w_end = calc_h_end(id,M->width,number_of_threads);
    reset_mem_allocated(id,mem_allocated,number_of_threads,mutex,cond,barrier);
    init2DMatrix_thread(OUT, M->width,M->height,id,mem_allocated,number_of_threads,mutex,cond,barrier);
    // M is a matrix of height 2, width 100, but now below for loop is iterating from i:0 .. 99, height and width messed up
    for(int i=w_start;i<=w_end;i++) {
        for(int j=0;j<M->height;j++) OUT->d[i][j] = M->d[j][i];
    }
    return 0;
}

int dotProduct_thread(TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* OUT,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    if (X->width != W->height) {
        return 1;
    }
    int h_start = calc_h_start(id,X->height,number_of_threads);
    int h_end = calc_h_end(id,X->height,number_of_threads);
    reset_mem_allocated(id,mem_allocated,number_of_threads,mutex,cond,barrier);
    init2DMatrix_thread(OUT,X->height,W->width,id,mem_allocated,number_of_threads,mutex,cond,barrier);
    for(int i=h_start;i<=h_end;i++) {
        for(int j=0;j<W->width;j++) {
            float sum = 0;
            for(int p=0;p<X->width;p++) sum += X->d[i][p]*W->d[p][j];
            OUT->d[i][j] = sum;
        }
    }
    return 0;
}

int elementwiseAdd2DMatrix_thread(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    if (A->height != B->height) return 1;
    if (A->width != B->width) return 1;
    int h_start = calc_h_start(id,A->height,number_of_threads);
    int h_end = calc_h_end(id,A->height,number_of_threads);
    reset_mem_allocated(id,mem_allocated,number_of_threads,mutex,cond,barrier);
    init2DMatrix_thread(OUT,A->height,A->width,id,mem_allocated,number_of_threads,mutex,cond,barrier);
    for(int i=h_start;i<=h_end;i++) {
        for(int j=0;j<A->width;j++) {
            OUT->d[i][j] = A->d[i][j] + B->d[i][j];
        }
    }
    return 0;
}

int elementwiseSub2DMatrix_thread(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    if (A->height != B->height) return 1;
    if (A->width != B->width) return 1;
    int h_start = calc_h_start(id,A->height,number_of_threads);
    int h_end = calc_h_end(id,A->height,number_of_threads);
    reset_mem_allocated(id,mem_allocated,number_of_threads,mutex,cond,barrier);
    init2DMatrix_thread(OUT,A->height,A->width,id,mem_allocated,number_of_threads,mutex,cond,barrier);
    for(int i=h_start;i<=h_end;i++) {
        for(int j=0;j<A->width;j++) {
            OUT->d[i][j] = A->d[i][j] - B->d[i][j];
        }
    }
    return 0;
}

int elementwiseMul2DMatrix_thread(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    if (A->height != B->height) return 1;
    if (A->width != B->width) return 1;
    int h_start = calc_h_start(id,A->height,number_of_threads);
    int h_end = calc_h_end(id,A->height,number_of_threads);
    reset_mem_allocated(id,mem_allocated,number_of_threads,mutex,cond,barrier);
    init2DMatrix_thread(OUT,A->height,A->width,id,mem_allocated,number_of_threads,mutex,cond,barrier);
    for(int i=h_start;i<=h_end;i++) {
        for(int j=0;j<A->width;j++) {
            OUT->d[i][j] = A->d[i][j] * B->d[i][j];
        }
    }
    return 0;
}

int elementwiseDiv2DMatrix_thread(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    if (A->height != B->height) return 1;
    if (A->width != B->width) return 1;
    int h_start = calc_h_start(id,A->height,number_of_threads);
    int h_end = calc_h_end(id,A->height,number_of_threads);
    reset_mem_allocated(id,mem_allocated,number_of_threads,mutex,cond,barrier);
    init2DMatrix_thread(OUT,A->height,A->width,id,mem_allocated,number_of_threads,mutex,cond,barrier);
    for(int i=h_start;i<=h_end;i++) {
        for(int j=0;j<A->width;j++) {
            OUT->d[i][j] = A->d[i][j] / B->d[i][j];
        }
    }
    return 0;
}

int elementSqrt_thread(TwoDMatrix* M,TwoDMatrix* OUT,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    int h_start = calc_h_start(id,M->height,number_of_threads);
    int h_end = calc_h_end(id,M->height,number_of_threads);
    reset_mem_allocated(id,mem_allocated,number_of_threads,mutex,cond,barrier);
    init2DMatrix_thread(OUT,M->height,M->width,id,mem_allocated,number_of_threads,mutex,cond,barrier);
    for(int i=h_start;i<=h_end;i++) {
        for(int j=0;j<M->width;j++) {
            OUT->d[i][j] = sqrt(M->d[i][j]);
        }
    }
    return 0;
}

int elementExp_thread(TwoDMatrix* M,TwoDMatrix* OUT,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    int h_start = calc_h_start(id,M->height,number_of_threads);
    int h_end = calc_h_end(id,M->height,number_of_threads);
    reset_mem_allocated(id,mem_allocated,number_of_threads,mutex,cond,barrier);
    init2DMatrix_thread(OUT,M->height,M->width,id,mem_allocated,number_of_threads,mutex,cond,barrier);
    for(int i=h_start;i<=h_end;i++) {
        for(int j=0;j<M->width;j++) {
            OUT->d[i][j] = exp(M->d[i][j]);
        }
    }
    return 0;
}

int elementLeakyReLU_thread(TwoDMatrix* M, float alpha,TwoDMatrix* OUT,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    int h_start = calc_h_start(id,M->height,number_of_threads);
    int h_end = calc_h_end(id,M->height,number_of_threads);
    reset_mem_allocated(id,mem_allocated,number_of_threads,mutex,cond,barrier);
    init2DMatrix_thread(OUT,M->height,M->width,id,mem_allocated,number_of_threads,mutex,cond,barrier);
    for(int i=h_start;i<=h_end;i++) {
        for(int j=0;j<M->width;j++) {
            if (M->d[i][j] >= 0) {
                OUT->d[i][j] = M->d[i][j];
            } else {
                OUT->d[i][j] = alpha * M->d[i][j];
            }
        }
    }
    return 0;
}

int elementAdd_thread(TwoDMatrix* M, float a,TwoDMatrix* OUT,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    int h_start = calc_h_start(id,M->height,number_of_threads);
    int h_end = calc_h_end(id,M->height,number_of_threads);
    reset_mem_allocated(id,mem_allocated,number_of_threads,mutex,cond,barrier);
    init2DMatrix_thread(OUT,M->height,M->width,id,mem_allocated,number_of_threads,mutex,cond,barrier);
    for(int i=h_start;i<=h_end;i++) {
        for(int j=0;j<M->width;j++) {
            OUT->d[i][j] = a+M->d[i][j];
        }
    }
    return 0;
}

int elementMul_thread(TwoDMatrix* M, float a,TwoDMatrix* OUT,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    int h_start = calc_h_start(id,M->height,number_of_threads);
    int h_end = calc_h_end(id,M->height,number_of_threads);
    reset_mem_allocated(id,mem_allocated,number_of_threads,mutex,cond,barrier);
    init2DMatrix_thread(OUT,M->height,M->width,id,mem_allocated,number_of_threads,mutex,cond,barrier);
    for(int i=h_start;i<=h_end;i++) {
        for(int j=0;j<M->width;j++) {
            OUT->d[i][j] = a*M->d[i][j];
        }
    }
    return 0;
}

int elementDiv_thread(TwoDMatrix* M, float a,TwoDMatrix* OUT,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    float n;
    if (a < 1e-6) {
        n = 1/(a+1e-6);
    } else {
        n = 1/a;
    }
    return elementMul_thread(M, n, OUT,id,mem_allocated,number_of_threads,mutex,cond,barrier);
    return 0;
}

int broadcastAdd_thread(TwoDMatrix* M, TwoDMatrix* b, int direction, TwoDMatrix* OUT, int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    int h_start = calc_h_start(id,M->height,number_of_threads);
    int h_end = calc_h_end(id,M->height,number_of_threads);
    reset_mem_allocated(id,mem_allocated,number_of_threads,mutex,cond,barrier);
    init2DMatrix_thread(OUT,M->height,M->width,id,mem_allocated,number_of_threads,mutex,cond,barrier);
    if (direction == 0) {
        // Add the column vector b horizontally
        for(int i=h_start;i<=h_end;i++) {
            for(int j=0;j<M->width;j++) {
                OUT->d[i][j] = M->d[i][j] + b->d[i][0];
            }
        }
    } else {
        // Add the row vector b vertically
        for(int i=h_start;i<=h_end;i++) {
            for(int j=0;j<M->width;j++) {
                OUT->d[i][j] = M->d[i][j] + b->d[0][j];
            }
        }
    }
    return 0;
}

int broadcastSub_thread(TwoDMatrix* M, TwoDMatrix* b, int direction, TwoDMatrix* OUT, int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    int h_start = calc_h_start(id,M->height,number_of_threads);
    int h_end = calc_h_end(id,M->height,number_of_threads);
    reset_mem_allocated(id,mem_allocated,number_of_threads,mutex,cond,barrier);
    init2DMatrix_thread(OUT,M->height,M->width,id,mem_allocated,number_of_threads,mutex,cond,barrier);
    if (direction == 0) {
        // Add the column vector b horizontally
        for(int i=h_start;i<=h_end;i++) {
            for(int j=0;j<M->width;j++) {
                OUT->d[i][j] = M->d[i][j] - b->d[i][0];
            }
        }
    } else {
        // Add the row vector b vertically
        for(int i=h_start;i<=h_end;i++) {
            for(int j=0;j<M->width;j++) {
                OUT->d[i][j] = M->d[i][j] - b->d[0][j];
            }
        }
    }
    return 0;
}

int broadcastMul_thread(TwoDMatrix* M, TwoDMatrix* b, int direction, TwoDMatrix* OUT, int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    int h_start = calc_h_start(id,M->height,number_of_threads);
    int h_end = calc_h_end(id,M->height,number_of_threads);
    reset_mem_allocated(id,mem_allocated,number_of_threads,mutex,cond,barrier);
    init2DMatrix_thread(OUT,M->height,M->width,id,mem_allocated,number_of_threads,mutex,cond,barrier);
    if (direction == 0) {
        // Add the column vector b horizontally
        for(int i=h_start;i<=h_end;i++) {
            for(int j=0;j<M->width;j++) {
                OUT->d[i][j] = M->d[i][j] * b->d[i][0];
            }
        }
    } else {
        // Add the row vector b vertically
        for(int i=h_start;i<=h_end;i++) {
            for(int j=0;j<M->width;j++) {
                OUT->d[i][j] = M->d[i][j] * b->d[0][j];
            }
        }
    }
    return 0;
}

int broadcastDiv_thread(TwoDMatrix* M, TwoDMatrix* b, int direction, TwoDMatrix* OUT, int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    int h_start = calc_h_start(id,M->height,number_of_threads);
    int h_end = calc_h_end(id,M->height,number_of_threads);
    reset_mem_allocated(id,mem_allocated,number_of_threads,mutex,cond,barrier);
    init2DMatrix_thread(OUT,M->height,M->width,id,mem_allocated,number_of_threads,mutex,cond,barrier);
    if (direction == 0) {
        // Add the column vector b horizontally
        for(int i=h_start;i<=h_end;i++) {
            for(int j=0;j<M->width;j++) {
                OUT->d[i][j] = M->d[i][j] / b->d[i][0];
            }
        }
    } else {
        // Add the row vector b vertically
        for(int i=h_start;i<=h_end;i++) {
            for(int j=0;j<M->width;j++) {
                OUT->d[i][j] = M->d[i][j] / b->d[0][j];
            }
        }
    }
    return 0;
}


int sumX2DMatrix_thread(TwoDMatrix* M,TwoDMatrix* OUT,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    int h_start = calc_h_start(id,M->height,number_of_threads);
    int h_end = calc_h_end(id,M->height,number_of_threads);
    reset_mem_allocated(id,mem_allocated,number_of_threads,mutex,cond,barrier);
    init2DMatrix_thread(OUT, M->height,1,id,mem_allocated,number_of_threads,mutex,cond,barrier);
    for(int i=h_start;i<=h_end;i++) {
        OUT->d[i][0] = 0;
        for(int j=0;j<M->width;j++) OUT->d[i][0] += M->d[i][j];
    }
    return 0;
}

int maxX2DMatrix_thread(TwoDMatrix* M,TwoDMatrix* OUT,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    int h_start = calc_h_start(id,M->height,number_of_threads);
    int h_end = calc_h_end(id,M->height,number_of_threads);
    reset_mem_allocated(id,mem_allocated,number_of_threads,mutex,cond,barrier);
    init2DMatrix_thread(OUT, M->height,1,id,mem_allocated,number_of_threads,mutex,cond,barrier);
    for(int i=h_start;i<=h_end;i++) {
        OUT->d[i][0] = M->d[i][0];
        for(int j=0;j<M->width;j++) OUT->d[i][0] = fmaxf(OUT->d[i][0], M->d[i][j]);
    }
    return 0;
}

///////////////////////////////////
// sumY2DMatrix_thread is only a single-threaded function 
// only thread 0 does the work, while other threads sit and watch
//////////////////////////////////
int sumY2DMatrix_thread(TwoDMatrix* M,TwoDMatrix* OUT,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    reset_mem_allocated(id,mem_allocated,number_of_threads,mutex,cond,barrier);
    if (id == 0) {
        pthread_mutex_lock(mutex);
        thread_barrier_init(barrier,number_of_threads);
        sumY2DMatrix(M,OUT);
        *mem_allocated = true;
        pthread_mutex_unlock(mutex);
    } else {
        pthread_mutex_lock(mutex);
        while(!(*mem_allocated)) pthread_cond_wait(cond,mutex);
        pthread_mutex_unlock(mutex);
    }
    if(id == 0) {
        pthread_mutex_lock(mutex);
        pthread_cond_broadcast(cond);
        pthread_mutex_unlock(mutex);
    }
    thread_barrier_wait_reinit(barrier,number_of_threads);
    return 0;
}

int maxY2DMatrix_thread(TwoDMatrix* M,TwoDMatrix* OUT,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    reset_mem_allocated(id,mem_allocated,number_of_threads,mutex,cond,barrier);
    if (id == 0) {
        pthread_mutex_lock(mutex);
        thread_barrier_init(barrier,number_of_threads);
        maxY2DMatrix(M,OUT);
        *mem_allocated = true;
        pthread_mutex_unlock(mutex);
    } else {
        pthread_mutex_lock(mutex);
        while(!(*mem_allocated)) pthread_cond_wait(cond,mutex);
        pthread_mutex_unlock(mutex);
    }
    if(id == 0) {
        pthread_mutex_lock(mutex);
        pthread_cond_broadcast(cond);
        pthread_mutex_unlock(mutex);
    }
    thread_barrier_wait_reinit(barrier,number_of_threads);
    return 0;
}

float sumAll_thread(TwoDMatrix* M,int id, bool* mem_allocated,int number_of_threads, pthread_mutex_t* mutex, pthread_cond_t* cond, thread_barrier_t* barrier) {
    reset_mem_allocated(id,mem_allocated,number_of_threads,mutex,cond,barrier);
    float* retval_ptr;
    char* share_memory_name = "/sumAll_thread_shared_memory";
    if (id == 0) {
        pthread_mutex_lock(mutex);
        thread_barrier_init(barrier,number_of_threads);
        retval_ptr = malloc(sizeof(float));
        *retval_ptr = sumAll(M);
        IPCWriteToSharedMem(share_memory_name,retval_ptr,sizeof(float));
        *mem_allocated = true;
        pthread_mutex_unlock(mutex);
    } else {
        pthread_mutex_lock(mutex);
        while(!(*mem_allocated)) pthread_cond_wait(cond,mutex);
        pthread_mutex_unlock(mutex);
    }
    if(id == 0) {
        pthread_mutex_lock(mutex);
        pthread_cond_broadcast(cond);
        pthread_mutex_unlock(mutex);
    }
    pthread_mutex_lock(mutex);
    IPCReadFromSharedMem(share_memory_name,retval_ptr,sizeof(float));
    pthread_mutex_unlock(mutex);
    float retval = *retval_ptr;
    thread_barrier_wait_reinit(barrier,number_of_threads);
    if (id == 0) {
        IPCRemoveSharedMemFile(share_memory_name);
        free(retval_ptr);
    }
    thread_barrier_wait_reinit(barrier,number_of_threads);
    return retval;
}
