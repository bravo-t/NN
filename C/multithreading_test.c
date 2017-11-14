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
#include "src/misc_utils.h"

struct DotProductRowArgs {
    TwoDMatrix* X;
    TwoDMatrix* W;
    TwoDMatrix* OUT;
    int h;
};
struct DotProductRowArgs_faster {
    TwoDMatrix* X;
    TwoDMatrix* W;
    TwoDMatrix* OUT;
    int h_start;
    int h_end;
};
int dotProduct_MT(TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* OUT, int number_of_threads);
void* dotProductRow(void* args);
int dotProduct_MT_faster(TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* OUT, int number_of_threads);
void* dotProductRow_faster(void* args);

int main(int argc, char const *argv[])
{
	TwoDMatrix* X = matrixMalloc(sizeof(TwoDMatrix));
	TwoDMatrix* M = matrixMalloc(sizeof(TwoDMatrix));
	init2DMatrixNormRand(X,10000,1000,0.0,1.0,2);
	init2DMatrixNormRand(M,1000,10000,0.0,1.0,2);
	TwoDMatrix* OUT_MT = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* OUT_ST = matrixMalloc(sizeof(TwoDMatrix)); 

    struct timeval t1,t2;
    double mt_time,st_time;
    gettimeofday(&t1,NULL);
    for(int i=0;i<1;i++) dotProduct_MT(X,M,OUT_MT,16);
    gettimeofday(&t2,NULL);
    mt_time = (t2.tv_sec - t1.tv_sec) * 1000.0;
    mt_time += (t2.tv_usec - t1.tv_usec) / 1000.0;
    
    gettimeofday(&t1,NULL);
    for(int i=0;i<1;i++) dotProduct_MT_faster(X,M,OUT_ST,16);
    gettimeofday(&t2,NULL);
    st_time = (t2.tv_sec - t1.tv_sec) * 1000.0;
    st_time += (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("Multithreaded dot took %f ms, optmized multithreaded dot took %f ms\n",mt_time,st_time );
	//dotProduct_MT(X,M,OUT,64);
    //checkMatrixDiff(OUT_MT,OUT_ST,1e-3);
	return 0;
}

int dotProduct_MT(TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* OUT, int number_of_threads) {
    if (X->width != W->height) {
        return 1;
    }
    init2DMatrix(OUT,X->height,W->width);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    for(int i=0;i<X->height;i++) {
        pthread_t* thread = malloc(sizeof(pthread_t)*number_of_threads);
        int t;
        for(;i%number_of_threads!=number_of_threads-1&&i<X->height;i++) {
            t = i%number_of_threads;
            struct DotProductRowArgs* thread_arg = malloc(sizeof(struct DotProductRowArgs));
            thread_arg->X = X;
            thread_arg->W = W;
            thread_arg->OUT = OUT;
            thread_arg->h = i;
            int create_error = pthread_create(&thread[t],&attr,dotProductRow,(void*) thread_arg);
            if (create_error) {
                printf("ERROR: Create thread failed.\n");
                exit(-1);
            }
        }
        void* status;
        for(int n=0;n<t;n++) {
            int join_error = pthread_join(thread[n],&status);
            if (join_error) {
                printf("ERROR: Join thread failed.\n");
                exit(-1);
            }
        }
    }
    return 0;
}

void* dotProductRow(void* args) {
    struct DotProductRowArgs* a = (struct DotProductRowArgs*) args;
    TwoDMatrix* X = a->X;
    TwoDMatrix* W = a->W;
    TwoDMatrix* OUT = a->OUT;
    int i = a->h;
    for(int j=0;j<W->width;j++) {
        float sum = 0;
        for(int p=0;p<X->width;p++) sum += X->d[i][p]*W->d[p][j];
        OUT->d[i][j] = sum;
    }
    free(args);
    pthread_exit(NULL);
}

int dotProduct_MT_faster(TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* OUT, int number_of_threads) {
    if (X->width != W->height) {
        return 1;
    }
    init2DMatrix(OUT,X->height,W->width);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_t* thread = malloc(sizeof(pthread_t)*number_of_threads);
    int H = X->height / number_of_threads + 1;
    if (number_of_threads > X->height) number_of_threads = X->height;
    int t = 0;
    for(;t<number_of_threads;t++) {
        struct DotProductRowArgs_faster* thread_arg = malloc(sizeof(struct DotProductRowArgs_faster));
        thread_arg->X = X;
        thread_arg->W = W;
        thread_arg->OUT = OUT;
        thread_arg->h_start = t*H;
        thread_arg->h_end = (t+1)*H-1;
        if (thread_arg->h_end >= X->height) thread_arg->h_end = X->height - 1;
        int create_error = pthread_create(&thread[t],&attr,dotProductRow_faster,(void*) thread_arg);
        if (create_error) {
            printf("ERROR: Create thread failed.\n");
            exit(-1);
        }
    }
    void* status;
    for(int n=0;n<t;n++) {
        int join_error = pthread_join(thread[n],&status);
        if (join_error) {
            printf("ERROR: Join thread failed.\n");
            exit(-1);
        }
    }        
    return 0;
}

void* dotProductRow_faster(void* args) {
    struct DotProductRowArgs_faster* a = (struct DotProductRowArgs_faster*) args;
    TwoDMatrix* X = a->X;
    TwoDMatrix* W = a->W;
    TwoDMatrix* OUT = a->OUT;
    for(int i=a->h_start;i<=a->h_end;i++) {
        for(int j=0;j<W->width;j++) {
            float sum = 0;
            for(int p=0;p<X->width;p++) sum += X->d[i][p]*W->d[p][j];
            OUT->d[i][j] = sum;
        }
    }
    free(args);
    pthread_exit(NULL);
}
