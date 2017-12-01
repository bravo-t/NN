#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>
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
    TwoDMatrix* XT = matrixMalloc_thread("/affineLayerBackword_thread_XT", id, sizeof(TwoDMatrix));
    TwoDMatrix* WT = matrixMalloc_thread("/affineLayerBackword_thread_XT", id, sizeof(TwoDMatrix));
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
    destroy2DMatrix_thread(XT,calc_h_start(id,XT->height),calc_h_end(id,XT->height),mem_allocated);
    destroy2DMatrix_thread(WT,calc_h_start(id,WT->height),calc_h_end(id,WT->height),mem_allocated);
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

int leakyReLUBackward_thread(TwoDMatrix* dM, TwoDMatrix* M, float alpha, TwoDMatrix* OUT) {
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

/////////////////////////////////////////////
// To be implemented
/////////////////////////////////////////////
float SVMLoss_thread(TwoDMatrix* score, TwoDMatrix* correct_label, TwoDMatrix* dscore, int id, bool* mem_allocated) {
    TwoDMatrix* margins = matrixMalloc_thread("/SVMLoss_thread_margins",sizeof(TwoDMatrix),id);
    int number_of_examples = score->height;
    int h_start = calc_h_start(number_of_examples, id);
    int h_end = calc_h_end(number_of_examples, id);
    reset_mem_allocated(id,mem_allocated);
    init2DMatrix_thread(margins, score->height, score->width, id, mem_allocated);
    reset_mem_allocated(id,mem_allocated);
    init2DMatrix_thread(dscore, score->height, score->width, id, mem_allocated);
    int* number_of_pos = calloc_thread("/SVMLoss_thread_number_of_examples", number_of_examples, sizeof(int), id);
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
    reset_mem_allocated(mem_allocated);
    return data_loss;
}

float softmaxLoss_thread(TwoDMatrix* score, TwoDMatrix* correct_label, TwoDMatrix* dscore,int id, bool* mem_allocated) {
    int h_start = calc_h_start(id,score->height);
    int h_end = calc_h_end(id,score->height);
    init2DMatrix_thread(dscore,score->height,score->width,id,mem_allocated);
    TwoDMatrix* max_scores = matrixMalloc_thread("/softmaxLoss_thread_max_cores_shm",sizeof(TwoDMatrix),id);
    init2DMatrix_thread(max_scores,score->height,1,id,mem_allocated);
    maxX2DMatrix_thread(score,max_scores,id,mem_allocated);
    TwoDMatrix* shifted = matrixMalloc_thread("/softmaxLoss_thread_shifted_shm",sizeof(TwoDMatrix),id);
    init2DMatrix_thread(shifted,score->height,score->width,id,mem_allocated);
    broadcastSub_thread(score,max_scores,0,shifted,id,mem_allocated);
    TwoDMatrix* exp_score = matrixMalloc_thread("/softmaxLoss_thread_exp_score_shm",sizeof(TwoDMatrix),id);
    init2DMatrix_thread(exp_score,score->height,score->width,id,mem_allocated);
    elementExp_thread(shifted,exp_score,id,mem_allocated);
    TwoDMatrix* exp_sum = matrixMalloc_thread("/softmaxLoss_thread_exp_sum_shm",sizeof(TwoDMatrix),id);
    init2DMatrix_thread(exp_sum,score->height,1,id,mem_allocated);
    sumX2DMatrix_thread(exp_score,exp_sum,id,mem_allocated);
    TwoDMatrix* probs = matrixMalloc_thread("/softmaxLoss_thread_probs_shm",sizeof(TwoDMatrix),id);
    init2DMatrix_thread(probs,score->height,score->width,id,mem_allocated);
    broadcastDiv_thread(exp_score,exp_sum,0,probs,id,mem_allocated);
    TwoDMatrix* correct_probs = matrixMalloc_thread("/softmaxLoss_thread_correct_probs_shm",sizeof(TwoDMatrix),id);
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

float L2RegLoss(TwoDMatrix** Ms,int network_depth, float reg_strength) {
    float reg_loss = 0;
    for (int i = 0; i < network_depth; i++) {
        TwoDMatrix* M_squared = matrixMalloc(sizeof(TwoDMatrix));
        init2DMatrix(M_squared,Ms[i]->height,Ms[i]->width);
        elementwiseMul2DMatrix(Ms[i],Ms[i],M_squared);
        reg_loss += 0.5*reg_strength*sumAll(M_squared);
        destroy2DMatrix(M_squared);
    }
    return reg_loss;
}

int L2RegLossBackward(TwoDMatrix* dM, TwoDMatrix* M, float reg_strength, TwoDMatrix* OUT) {
    init2DMatrix(OUT, M->height, M->width);
    TwoDMatrix* M_scaled = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(M_scaled, M->height, M->width);
    elementMul(M,reg_strength,M_scaled);
    int retval = elementwiseAdd2DMatrix(dM, M_scaled, OUT);
    destroy2DMatrix(M_scaled);
    return retval;
}

int momentumUpdate(TwoDMatrix* X, TwoDMatrix* dX, TwoDMatrix* v, float mu, float learning_rate,  TwoDMatrix* OUT) {
    init2DMatrix(OUT, X->height, X->width);
    TwoDMatrix* v_mu = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(v_mu, v->height, v->width);
    elementMul(v, mu, v_mu);
    TwoDMatrix* dX_scaled = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(dX_scaled, X->height, X->width);
    elementMul(dX,learning_rate,dX_scaled);
    int retval = elementwiseSub2DMatrix(v_mu,dX_scaled,v);
    destroy2DMatrix(v_mu);
    destroy2DMatrix(dX_scaled);
    retval += elementwiseAdd2DMatrix(X,v,OUT);
    return retval;
}

int NAGUpdate(TwoDMatrix* X, TwoDMatrix* dX, TwoDMatrix* v, TwoDMatrix* v_prev, float mu, float learning_rate,  TwoDMatrix* OUT) {
    init2DMatrix(OUT, X->height, X->width);
    copyTwoDMatrix(v,v_prev);
    TwoDMatrix* v_mu = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(v_mu, v->height, v->width);
    elementMul(v, mu, v_mu);
    TwoDMatrix* dX_scaled = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(dX_scaled, X->height, X->width);
    elementMul(dX,learning_rate,dX_scaled);
    elementwiseSub2DMatrix(v_mu,dX_scaled,v);
    destroy2DMatrix(v_mu);
    destroy2DMatrix(dX_scaled);
    TwoDMatrix* v_prev_mu = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(v_prev_mu, v->height, v->width);
    elementMul(v_prev, -1*mu, v_prev_mu);
    TwoDMatrix* v_mu_plus_one = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(v_mu_plus_one, v->height, v->width);
    elementMul(v, 1+mu, v_mu_plus_one);
    TwoDMatrix* X_update = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(X_update, v->height, v->width);
    elementwiseAdd2DMatrix(X,X_update,OUT);
    destroy2DMatrix(v_prev_mu);
    destroy2DMatrix(v_mu_plus_one);
    destroy2DMatrix(X_update);
    return 0;
}

int RMSProp(TwoDMatrix* X, TwoDMatrix* dX, TwoDMatrix* cache, float learning_rate, float decay_rate, float eps, TwoDMatrix* OUT) {
    TwoDMatrix* cache_scaled = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(cache_scaled, X->height, X->width);
    elementMul(cache,decay_rate,cache_scaled);
    TwoDMatrix* dX_squared = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(dX_squared, dX->height, dX->width);
    elementwiseMul2DMatrix(dX,dX,dX_squared);
    TwoDMatrix* dX_squared_scaled = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(dX_squared_scaled, dX->height, dX->width);
    elementMul(dX_squared,1-decay_rate,dX_squared_scaled);
    elementwiseAdd2DMatrix(cache_scaled,dX_squared_scaled,cache);
    destroy2DMatrix(cache_scaled);
    destroy2DMatrix(dX_squared);
    destroy2DMatrix(dX_squared_scaled);
    TwoDMatrix* cache_sqrt = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(cache_sqrt, X->height, X->width);
    elementSqrt(cache,cache_sqrt);
    TwoDMatrix* cache_sqrt_eps = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(cache_sqrt_eps, X->height, X->width);
    elementAdd(cache_sqrt, eps, cache_sqrt_eps);
    TwoDMatrix* dX_scaled = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(dX_scaled, X->height, X->width);
    elementMul(dX,learning_rate,dX_scaled);
    TwoDMatrix* X_update = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(X_update, dX->height, dX->width);
    elementwiseDiv2DMatrix(dX_scaled,cache_sqrt_eps,X_update);
    elementwiseSub2DMatrix(X,X_update,OUT);
    destroy2DMatrix(cache_sqrt);
    destroy2DMatrix(cache_sqrt_eps);
    destroy2DMatrix(dX_scaled);
    destroy2DMatrix(X_update);
    return 0;
}

int batchnorm_training_forward(TwoDMatrix* M, float momentum, float eps, TwoDMatrix* gamma, TwoDMatrix* beta, TwoDMatrix* OUT, TwoDMatrix* mean_cache, TwoDMatrix* var_cache, TwoDMatrix* mean, TwoDMatrix* var, TwoDMatrix* M_normalized) {
    matrixYMeanVar(M, mean, var);
    TwoDMatrix* M_centered = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(M_centered,M->height,M->width);
    broadcastSub(M,mean,1,M_centered);
    TwoDMatrix* var_eps = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(var_eps,var->height,var->width);
    TwoDMatrix* std_var = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(std_var,var->height,var->width);
    elementAdd(var,eps,var_eps);
    elementSqrt(var_eps,std_var);
    init2DMatrix(M_normalized,M->height,M->width);
    broadcastDiv(M_centered,std_var,1,M_normalized);
    init2DMatrix(OUT, M->height, M->width);
    broadcastMul(M_normalized,gamma,1,M_normalized);
    broadcastAdd(M_normalized,beta,1,OUT);
    TwoDMatrix* mean_scaled = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(mean_scaled,mean->height,mean->width);
    TwoDMatrix* var_scaled = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(var_scaled,var->height,var->width);
    elementMul(mean_cache,momentum,mean_cache);
    elementMul(mean,1-momentum,mean_scaled);
    elementwiseAdd2DMatrix(mean_cache,mean_scaled,mean_cache);
    elementMul(var_cache,momentum,var_cache);
    elementMul(var,1-momentum,var_scaled);
    elementwiseAdd2DMatrix(var_cache,var_scaled,var_cache);
    destroy2DMatrix(M_centered);
    destroy2DMatrix(var_eps);
    destroy2DMatrix(std_var);
    destroy2DMatrix(mean_scaled);
    destroy2DMatrix(var_scaled);
    return 0;
}

int batchnorm_test_forward(TwoDMatrix* M, TwoDMatrix* mean_cache, TwoDMatrix* var_cache, float eps, TwoDMatrix* gamma, TwoDMatrix* beta, TwoDMatrix* OUT) {
    TwoDMatrix* M_centered = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(M_centered,M->height,M->width);
    broadcastSub(M,mean_cache,1,M_centered);
    TwoDMatrix* var_eps = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(var_eps,var_cache->height,var_cache->width);
    TwoDMatrix* std_var = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(std_var,var_cache->height,var_cache->width);
    elementAdd(var_cache,eps,var_eps);
    elementSqrt(var_eps,std_var);
    TwoDMatrix* M_normalized = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(M_normalized,M->height,M->width);
    broadcastDiv(M_centered,std_var,1,M_normalized);
    init2DMatrix(OUT, M->height, M->width);
    broadcastMul(M_normalized,gamma,1,M_normalized);
    broadcastAdd(M_normalized,beta,1,OUT);
    destroy2DMatrix(M_centered);
    destroy2DMatrix(var_eps);
    destroy2DMatrix(std_var);
    destroy2DMatrix(M_normalized);
    return 0;
}

int batchnorm_backward(TwoDMatrix* dOUT, TwoDMatrix* M, TwoDMatrix* M_normalized, TwoDMatrix* gamma, TwoDMatrix* beta, TwoDMatrix* mean, TwoDMatrix* var, float eps, TwoDMatrix* dM, TwoDMatrix* dgamma, TwoDMatrix* dbeta) {
    init2DMatrix(dbeta,beta->height,beta->width);
    sumY2DMatrix(dOUT,dbeta);
    init2DMatrix(dgamma,gamma->height,gamma->width);
    TwoDMatrix* dOUT_M_normalized = matrixMalloc(sizeof(TwoDMatrix));
    elementwiseMul2DMatrix(dOUT, M_normalized,dOUT_M_normalized);
    sumY2DMatrix(dOUT_M_normalized,dgamma);
    TwoDMatrix* dM_normalized = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(dM_normalized, M_normalized->height, M_normalized->width);
    broadcastMul(dOUT,gamma,1,dM_normalized);
    TwoDMatrix* M_mu = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(M_mu, M_normalized->height, M_normalized->width);
    broadcastSub(M,mean,1,M_mu);
    TwoDMatrix* std_var = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(std_var, var->height, var->width);
    elementAdd(var, eps, std_var);
    elementSqrt(std_var,std_var);
    TwoDMatrix* std_var_inv = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(std_var_inv, var->height, var->width);
    for(int i=0;i<std_var_inv->height;i++) {
        for(int j=0;j<std_var_inv->width;j++) {
            std_var_inv->d[i][j] = 1.0/std_var->d[i][j];
        }
    }
    TwoDMatrix* std_var_inv_cube = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(std_var_inv_cube, var->height, var->width);
    elementwiseMul2DMatrix(std_var_inv,std_var_inv,std_var_inv_cube);
    elementwiseMul2DMatrix(std_var_inv_cube,std_var_inv,std_var_inv_cube);
    TwoDMatrix* M_tmp = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(M_tmp,M->height,M->width);
    elementwiseMul2DMatrix(dM_normalized,M_mu,M_tmp);
    TwoDMatrix* dvar = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(dvar, var->height, var->width);
    sumY2DMatrix(M_tmp,dvar);
    elementMul(dvar,-0.5,dvar);
    elementwiseMul2DMatrix(dvar,std_var_inv_cube,dvar);
    TwoDMatrix* M_mu_mean = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(M_mu_mean,mean->height,mean->width);
    matrixYMeanVar(M_mu, M_mu_mean, NULL);
    TwoDMatrix* var_tmp = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(var_tmp,var->height,var->width);
    elementwiseMul2DMatrix(dvar,M_mu_mean,var_tmp);
    elementMul(var_tmp,2.0,var_tmp);
    TwoDMatrix* dmean = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(dmean, mean->height, mean->width);
    TwoDMatrix* M_tmp2 = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(M_tmp2,M->height,M->width);
    broadcastMul(dM_normalized,std_var_inv,1,M_tmp2);
    TwoDMatrix* mean_tmp = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(mean_tmp,mean->height,mean->width);
    sumY2DMatrix(M_tmp2,mean_tmp);
    elementMul(M_tmp2,-1.0,M_tmp2);
    elementwiseSub2DMatrix(M_tmp2,var_tmp,dmean);
    TwoDMatrix* dM1 = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(dM1,M->height,M->width);
    broadcastMul(dM_normalized,std_var_inv,1,dM1);
    TwoDMatrix* dM2 = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(dM2,M->height,M->width);
    broadcastMul(M_mu,dvar,1,dM2);
    elementMul(dM2,2.0,dM2);
    elementDiv(dM2,dM2->height,dM2);
    elementwiseAdd2DMatrix(dM1,dM2,dM);
    TwoDMatrix* dmean_tmp = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(dmean_tmp,dmean->height,dmean->width);
    elementDiv(dmean,M->height,dmean_tmp);
    broadcastAdd(dM,dmean_tmp,1,dM);
    destroy2DMatrix(dM_normalized);
    destroy2DMatrix(M_mu);
    destroy2DMatrix(std_var);
    destroy2DMatrix(std_var_inv);
    destroy2DMatrix(std_var_inv_cube);
    destroy2DMatrix(M_tmp);
    destroy2DMatrix(dvar);
    destroy2DMatrix(M_mu_mean);
    destroy2DMatrix(var_tmp);
    destroy2DMatrix(dmean);
    destroy2DMatrix(M_tmp2);
    destroy2DMatrix(mean_tmp);
    destroy2DMatrix(dM1);
    destroy2DMatrix(dM2);
    destroy2DMatrix(dmean_tmp);
    destroy2DMatrix(dOUT_M_normalized);
    return 0;
}
