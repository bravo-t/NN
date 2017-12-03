#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>
#include <semaphore.h>
#include "network_type.h"
#include "inter-process_communication.h"
#include "matrix_operations.h"
#include "matrix_operations_multithread.h"
#include "layers.h"
#include "layers_multithread.h"

int affineLayerForward_thread(TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* b, TwoDMatrix* OUT, int id, bool* mem_allocated) {
    //int h_start = calc_h_start(id,X->height);
    //int h_end = calc_h_end(id,X->height);
    //reset_mem_allocated(mem_allocated);
    //init2DMatrix_thread(OUT, X->height, W->width, id, mem_allocated);
    if (dotProduct_thread(X,W,OUT,id,mem_allocated)) {
        printf("ERROR: Input matrix size mismatch: X->width = %d, W->height = %d\n", X->width,W->height);
        exit(1);
    }
    broadcastAdd_thread(OUT, b, 1, OUT,id,mem_allocated);
    return 0;
}

int affineLayerBackword_thread(TwoDMatrix* dOUT, TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* b, TwoDMatrix* dX, TwoDMatrix* dW, TwoDMatrix* db,int id, bool* mem_allocated) {
    //init2DMatrix(dX, X->height, X->width);
    //init2DMatrix(dW, W->height, W->width);
    //init2DMatrix(db, b->height, b->width);
    TwoDMatrix* XT = matrixMalloc_thread("/affineLayerBackword_thread_XT", sizeof(TwoDMatrix), id,mem_allocated);
    TwoDMatrix* WT = matrixMalloc_thread("/affineLayerBackword_thread_XT", sizeof(TwoDMatrix), id,mem_allocated);
    //init2DMatrix(XT, X->width, X->height);
    //init2DMatrix(WT, W->width, W->height);
    transpose2DMatrix_thread(X, XT, id, mem_allocated);
    transpose2DMatrix_thread(W, WT, id, mem_allocated);
    if (dotProduct_thread(dOUT,WT,dX,id,mem_allocated)) {
        printf("ERROR: Input matrix size mismatch: dOUT->width = %d, W.T->height = %d\n", dOUT->width,WT->height);
        exit(1);
    }
    if (dotProduct_thread(XT,dOUT,dW,id,mem_allocated)) {
        printf("ERROR: Input matrix size mismatch: X.T->width = %d, dOUT->height = %d\n", XT->width,dOUT->height);
        exit(1);
    }
    if (db->height == 1) {
        sumY2DMatrix_thread(dOUT,db,id,mem_allocated);
    } else {
        sumX2DMatrix_thread(dOUT,db,id,mem_allocated);
    }
    destroy2DMatrix_thread(XT,id,mem_allocated);
    destroy2DMatrix_thread(WT,id,mem_allocated);
    return 0;
}

int leakyReLUForward_thread(TwoDMatrix* M, float alpha, TwoDMatrix* OUT,int id, bool* mem_allocated) {
    return elementLeakyReLU_thread(M, alpha, OUT, id, mem_allocated);
}

int vanillaUpdate_thread(TwoDMatrix* M, TwoDMatrix* dM, float learning_rate, TwoDMatrix* OUT, int id, bool* mem_allocated) {
    int h_start = calc_h_start(id,M->height);
    int h_end = calc_h_end(id,M->height);
    reset_mem_allocated(id,mem_allocated);
    init2DMatrix_thread(OUT,M->height,M->width,id,mem_allocated);
    for(int i=h_start;i<=h_end;i++) {
        for(int j=0;j<M->width;j++) {
            OUT->d[i][j] = M->d[i][j] - dM->d[i][j]*learning_rate;
/*
#if defined(DEBUG) && DEBUG > 3
            if (isnan(OUT->d[i][j])) {
                printf("DEBUG: vanillaUpdate produced a nan: %f - %f * %f = %f\n",
                    M->d[i][j],
                    dM->d[i][j],
                    learning_rate,
                    OUT->d[i][j]);
            }
#endif
*/
        }
    }
    /*
    init2DMatrix(OUT, M->height, M->width);
    TwoDMatrix* dM_scaled = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(dM_scaled, M->height, M->width);
    elementMul(dM,learning_rate,dM_scaled);
    int retval = elementwiseSub2DMatrix(M, dM_scaled, OUT);
    destroy2DMatrix(dM_scaled);
    */
    return 0;
}

int leakyReLUBackward_thread(TwoDMatrix* dM, TwoDMatrix* M, float alpha, TwoDMatrix* OUT,int id, bool* mem_allocated) {
    int h_start = calc_h_start(id,dM->height);
    int h_end = calc_h_end(id,dM->height);
    reset_mem_allocated(id,mem_allocated);
    init2DMatrix_thread(OUT,dM->height,dM->width,id,mem_allocated);
    for(int i=h_start;i<=h_end;i++) {
        for(int j=0;j<dM->width;j++) {
            if (M->d[i][j] > 0) {
                OUT->d[i][j] = dM->d[i][j];
            } else {
                OUT->d[i][j] = alpha * dM->d[i][j];
            }
        }
    }
    return 0;
}

float SVMLoss_thread(TwoDMatrix* score, TwoDMatrix* correct_label, TwoDMatrix* dscore, int id, bool* mem_allocated) {
    TwoDMatrix* margins = matrixMalloc_thread("/SVMLoss_thread_margins",sizeof(TwoDMatrix),id,mem_allocated);
    int number_of_examples = score->height;
    int h_start = calc_h_start(number_of_examples, id);
    int h_end = calc_h_end(number_of_examples, id);
    reset_mem_allocated(id,mem_allocated);
    init2DMatrix_thread(margins, score->height, score->width, id, mem_allocated);
    reset_mem_allocated(id,mem_allocated);
    init2DMatrix_thread(dscore, score->height, score->width, id, mem_allocated);
    int* number_of_pos = calloc_thread("/SVMLoss_thread_number_of_examples", number_of_examples, sizeof(int), id,mem_allocated);
    for(int i=h_start;i<=h_end;i++) {
        int correct_index = correct_label->d[i][0];
        float correct_score = score->d[i][correct_index];
        for(int j=0;j!=correct_index&&j<score->width;j++) {
            margins->d[i][j] = fmaxf(0,score->d[i][j] - correct_score + 1);
            if (margins->d[i][j] > 0) {
                number_of_pos[i]++;
                dscore->d[i][j] = 1;
            } else {
                dscore->d[i][j] = 0;
            }
        }
        margins->d[i][correct_index] = 0;
    }
    float data_loss = sumAll_thread(margins,id,mem_allocated) / number_of_examples;
    for(int i=0;i<score->height;i++) {
        int correct_index = correct_label->d[i][0];
        dscore->d[i][correct_index] -= number_of_pos[i];
    }
    elementDiv_thread(dscore,number_of_examples,dscore,id,mem_allocated);
    destroy2DMatrix_thread(margins,id,mem_allocated);
    if(id == 0) free(number_of_pos);
    // Here the reset_mem_allocated function is used as a thread barrier
    reset_mem_allocated(id,mem_allocated);
    return data_loss;
}

float softmaxLoss_thread(TwoDMatrix* score, TwoDMatrix* correct_label, TwoDMatrix* dscore,int id, bool* mem_allocated) {
    int h_start = calc_h_start(id,score->height);
    int h_end = calc_h_end(id,score->height);
    init2DMatrix_thread(dscore,score->height,score->width,id,mem_allocated);
    TwoDMatrix* max_scores = matrixMalloc_thread("/softmaxLoss_thread_max_cores_shm",sizeof(TwoDMatrix),id,mem_allocated);
    init2DMatrix_thread(max_scores,score->height,1,id,mem_allocated);
    maxX2DMatrix_thread(score,max_scores,id,mem_allocated);
    TwoDMatrix* shifted = matrixMalloc_thread("/softmaxLoss_thread_shifted_shm",sizeof(TwoDMatrix),id,mem_allocated);
    init2DMatrix_thread(shifted,score->height,score->width,id,mem_allocated);
    broadcastSub_thread(score,max_scores,0,shifted,id,mem_allocated);
    TwoDMatrix* exp_score = matrixMalloc_thread("/softmaxLoss_thread_exp_score_shm",sizeof(TwoDMatrix),id,mem_allocated);
    init2DMatrix_thread(exp_score,score->height,score->width,id,mem_allocated);
    elementExp_thread(shifted,exp_score,id,mem_allocated);
    TwoDMatrix* exp_sum = matrixMalloc_thread("/softmaxLoss_thread_exp_sum_shm",sizeof(TwoDMatrix),id,mem_allocated);
    init2DMatrix_thread(exp_sum,score->height,1,id,mem_allocated);
    sumX2DMatrix_thread(exp_score,exp_sum,id,mem_allocated);
    TwoDMatrix* probs = matrixMalloc_thread("/softmaxLoss_thread_probs_shm",sizeof(TwoDMatrix),id,mem_allocated);
    init2DMatrix_thread(probs,score->height,score->width,id,mem_allocated);
    broadcastDiv_thread(exp_score,exp_sum,0,probs,id,mem_allocated);
    TwoDMatrix* correct_probs = matrixMalloc_thread("/softmaxLoss_thread_correct_probs_shm",sizeof(TwoDMatrix),id,mem_allocated);
    init2DMatrix_thread(correct_probs,score->height,1,id,mem_allocated);
    for(int i=h_start;i<=h_end;i++) {
        int correct_index = correct_label->d[i][0];
        for(int j=0;j<score->width;j++) {
            dscore->d[i][j] = probs->d[i][j];
            if(j == correct_index) {
                dscore->d[i][j] -= 1;
            }
        }
        if (probs->d[i][correct_index] >= 1e-6) {
            correct_probs->d[i][0] = -log(probs->d[i][correct_index]);
        } else {
            // log(0) will produce a nan, which will break the network. Add a small number to fix it
            correct_probs->d[i][0] = -log(probs->d[i][correct_index]+1e-6);
        }
    }
    //printf("correct_probs = \n");
    //printMatrix(correct_probs);
    int number_of_examples = score->height;
    float data_loss = sumAll_thread(correct_probs,id,mem_allocated) / number_of_examples;
    destroy2DMatrix_thread(max_scores,id,mem_allocated);
    destroy2DMatrix_thread(shifted,id,mem_allocated);
    destroy2DMatrix_thread(exp_score,id,mem_allocated);
    destroy2DMatrix_thread(exp_sum,id,mem_allocated);
    destroy2DMatrix_thread(probs,id,mem_allocated);
    destroy2DMatrix_thread(correct_probs,id,mem_allocated);
    return data_loss;
}

int L2RegLossBackward_thread(TwoDMatrix* dM, TwoDMatrix* M, float reg_strength, TwoDMatrix* OUT, int id, bool* mem_allocated) {
    TwoDMatrix* M_scaled = matrixMalloc_thread("/L2RegLossBackward_thread_M_scaled_shm",sizeof(TwoDMatrix),id,mem_allocated);
    init2DMatrix_thread(M_scaled, M->height, M->width,id,mem_allocated);
    elementMul_thread(M,reg_strength,M_scaled,id,mem_allocated);
    int retval = elementwiseAdd2DMatrix_thread(dM, M_scaled, OUT,id,mem_allocated);
    destroy2DMatrix_thread(M_scaled,id,mem_allocated);
    return retval;
}

int RMSProp_thread(TwoDMatrix* X, TwoDMatrix* dX, TwoDMatrix* cache, float learning_rate, float decay_rate, float eps, TwoDMatrix* OUT,int id, bool* mem_allocated) {
    TwoDMatrix* cache_scaled = matrixMalloc_thread("/RMSProp_thread_cache_scaled_shm",sizeof(TwoDMatrix),id,mem_allocated);
    elementMul_thread(cache,decay_rate,cache_scaled,id,mem_allocated);
    TwoDMatrix* dX_squared = matrixMalloc_thread("/RMSProp_thread_dX_squared_shm",sizeof(TwoDMatrix),id,mem_allocated);
    elementwiseMul2DMatrix_thread(dX,dX,dX_squared,id,mem_allocated);
    TwoDMatrix* dX_squared_scaled = matrixMalloc_thread("/RMSProp_thread_dX_squared_scaled_shm",sizeof(TwoDMatrix),id,mem_allocated);
    elementMul_thread(dX_squared,1-decay_rate,dX_squared_scaled,id,mem_allocated);
    elementwiseAdd2DMatrix_thread(cache_scaled,dX_squared_scaled,cache,id,mem_allocated);
    destroy2DMatrix_thread(cache_scaled,id,mem_allocated);
    destroy2DMatrix_thread(dX_squared,id,mem_allocated);
    destroy2DMatrix_thread(dX_squared_scaled,id,mem_allocated);
    TwoDMatrix* cache_sqrt = matrixMalloc_thread("/RMSProp_thread_cache_sqrt_shm",sizeof(TwoDMatrix),id,mem_allocated);
    elementSqrt_thread(cache,cache_sqrt,id,mem_allocated);
    TwoDMatrix* cache_sqrt_eps = matrixMalloc_thread("/RMSProp_thread_cache_sqrt_eps_shm",sizeof(TwoDMatrix),id,mem_allocated);
    elementAdd_thread(cache_sqrt, eps, cache_sqrt_eps,id,mem_allocated);
    TwoDMatrix* dX_scaled = matrixMalloc_thread("/RMSProp_thread_dX_scaled_shm",sizeof(TwoDMatrix),id,mem_allocated);
    elementMul_thread(dX,learning_rate,dX_scaled,id,mem_allocated);
    TwoDMatrix* X_update = matrixMalloc_thread("/RMSProp_thread_X_update_shm",sizeof(TwoDMatrix),id,mem_allocated);
    elementwiseDiv2DMatrix_thread(dX_scaled,cache_sqrt_eps,X_update,id,mem_allocated);
    elementwiseSub2DMatrix_thread(X,X_update,OUT,id,mem_allocated);
    destroy2DMatrix_thread(cache_sqrt,id,mem_allocated);
    destroy2DMatrix_thread(cache_sqrt_eps,id,mem_allocated);
    destroy2DMatrix_thread(dX_scaled,id,mem_allocated);
    destroy2DMatrix_thread(X_update,id,mem_allocated);
    return 0;
}