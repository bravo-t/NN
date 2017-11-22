#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <pthread.h>
#include "network_type.h"
#include "matrix_operations.h"
#include "matrix_operations_multithread.h"
#include "layers_multithread.h"

int affineLayerForward_MT(TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* b, TwoDMatrix* OUT, int number_of_threads) {
    init2DMatrix_MT(OUT, X->height, W->width, number_of_threads);
    if (dotProduct_MT(X,W,OUT, number_of_threads)) {
        printf("ERROR: Input matrix size mismatch: X->width = %d, W->height = %d\n", X->width,W->height);
        exit(1);
    }
    broadcastAdd_MT(OUT, b, 1, OUT, number_of_threads);
    return 0;
}

int affineLayerBackword_MT(TwoDMatrix* dOUT, TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* b, TwoDMatrix* dX, TwoDMatrix* dW, TwoDMatrix* db, int number_of_threads) {
    init2DMatrix_MT(dX, X->height, X->width, number_of_threads);
    init2DMatrix_MT(dW, W->height, W->width, number_of_threads);
    init2DMatrix_MT(db, b->height, b->width, number_of_threads);
    TwoDMatrix* XT = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* WT = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(XT, X->width, X->height, number_of_threads);
    init2DMatrix_MT(WT, W->width, W->height, number_of_threads);
    transpose2DMatrix_MT(X, XT, number_of_threads);
    transpose2DMatrix_MT(W, WT, number_of_threads);
    if (dotProduct_MT(dOUT,WT,dX, number_of_threads)) {
        printf("ERROR: Input matrix size mismatch: dOUT->width = %d, W.T->height = %d\n", dOUT->width,WT->height);
        exit(1);
    }
    if (dotProduct_MT(XT,dOUT,dW, number_of_threads)) {
        printf("ERROR: Input matrix size mismatch: X.T->width = %d, dOUT->height = %d\n", XT->width,dOUT->height);
        exit(1);
    }
    if (db->height == 1) {
        sumY2DMatrix(dOUT,db);
    } else {
        sumX2DMatrix(dOUT,db);
    }
    destroy2DMatrix_MT(XT, number_of_threads);
    destroy2DMatrix_MT(WT, number_of_threads);
    return 0;
}

int leakyReLUForward_MT(TwoDMatrix* M, float alpha, TwoDMatrix* OUT, int number_of_threads) {
    return elementLeakyReLU_MT(M, alpha, OUT, number_of_threads);
}

int vanillaUpdate_MT(TwoDMatrix* M, TwoDMatrix* dM, float learning_rate, TwoDMatrix* OUT, int number_of_threads) {
    init2DMatrix_MT(OUT, M->height, M->width,number_of_threads);
    TwoDMatrix* dM_scaled = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(dM_scaled, M->height, M->width,number_of_threads);
    elementMul_MT(dM,learning_rate,dM_scaled,number_of_threads);
    int retval = elementwiseSub2DMatrix_MT(M, dM_scaled, OUT,number_of_threads);
    destroy2DMatrix_MT(dM_scaled,number_of_threads);
    return retval;
}

void* leakyReLUBackwardRow(void* args) {
    TwoDMatrixOperationsRowArgs* a = (TwoDMatrixOperationsRowArgs*) args;
    TwoDMatrix* dM = a->M;
    TwoDMatrix* M = a->X;
    TwoDMatrix* OUT = a->OUT;
    float alpha = a->f;
    for(int i=a->h_start;i<=a->h_end;i++) {
        for(int j=0;j<dM->width;j++) {
            if (M->d[i][j] >= 0) {
                OUT->d[i][j] = dM->d[i][j];
            } else {
                OUT->d[i][j] = alpha * dM->d[i][j];
            }
        }
    }
    free(args);
    pthread_exit(NULL);
}

int leakyReLUBackward_MT(TwoDMatrix* dM, TwoDMatrix* M, float alpha, TwoDMatrix* OUT, int number_of_threads) {
    TwoDMatrixOperationsRowArgs* thread_arg = malloc(sizeof(TwoDMatrixOperationsRowArgs));
    thread_arg->M = dM;
    thread_arg->X = M;
    thread_arg->OUT = OUT;
    thread_arg->f = alpha;
    twoDMatrixOperationMultithreadWrapper(thread_arg,
        M->height, 
        M->height, 
        M->width, 
        leakyReLUBackwardRow,
        number_of_threads);
    free(thread_arg);
    return 0;
}

float softmaxLoss_MT(TwoDMatrix* score, TwoDMatrix* correct_label, TwoDMatrix* dscore, int number_of_threads) {
    init2DMatrix_MT(dscore,score->height,score->width,number_of_threads);
    TwoDMatrix* max_scores = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(max_scores,score->height,1,number_of_threads);
    maxX2DMatrix(score,max_scores);
    //printf("score = \n");
    //printMatrix(score);
    //printf("max_scores = \n");
    //printMatrix(max_scores);
    TwoDMatrix* shifted = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(shifted,score->height,score->width,number_of_threads);
    broadcastSub_MT(score,max_scores,0,shifted,number_of_threads);
    //printf("shifted = \n");
    //printMatrix(shifted);
    TwoDMatrix* exp_score = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(exp_score,score->height,score->width,number_of_threads);
    elementExp_MT(shifted,exp_score,number_of_threads);
    //printf("exp_score = \n");
    //printMatrix(exp_score);
    TwoDMatrix* exp_sum = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(exp_sum,score->height,1,number_of_threads);
    sumX2DMatrix(exp_score,exp_sum);
    //printf("exp_sum = \n");
    //printMatrix(exp_sum);
    TwoDMatrix* probs = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(probs,score->height,score->width,number_of_threads);
    broadcastDiv_MT(exp_score,exp_sum,0,probs,number_of_threads);
    //printf("probs = \n");
    //printMatrix(probs);
    TwoDMatrix* correct_probs = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(correct_probs,score->height,1,number_of_threads);
    for(int i=0;i<score->height;i++) {
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
    float data_loss = sumAll(correct_probs) / number_of_examples;
    destroy2DMatrix_MT(max_scores,number_of_threads);
    destroy2DMatrix_MT(shifted,number_of_threads);
    destroy2DMatrix_MT(exp_score,number_of_threads);
    destroy2DMatrix_MT(exp_sum,number_of_threads);
    destroy2DMatrix_MT(probs,number_of_threads);
    destroy2DMatrix_MT(correct_probs,number_of_threads);
    return data_loss;
}

float L2RegLoss_MT(TwoDMatrix** Ms,int network_depth, float reg_strength, int number_of_threads) {
    float reg_loss = 0;
    for (int i = 0; i < network_depth; i++) {
        TwoDMatrix* M_squared = matrixMalloc(sizeof(TwoDMatrix));
        init2DMatrix_MT(M_squared,Ms[i]->height,Ms[i]->width,number_of_threads);
        elementwiseMul2DMatrix_MT(Ms[i],Ms[i],M_squared,number_of_threads);
        reg_loss += 0.5*reg_strength*sumAll(M_squared);
        destroy2DMatrix_MT(M_squared,number_of_threads);
    }
    return reg_loss;
}

int L2RegLossBackward_MT(TwoDMatrix* dM, TwoDMatrix* M, float reg_strength, TwoDMatrix* OUT, int number_of_threads) {
    init2DMatrix_MT(OUT, M->height, M->width,number_of_threads);
    TwoDMatrix* M_scaled = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(M_scaled, M->height, M->width,number_of_threads);
    elementMul_MT(M,reg_strength,M_scaled,number_of_threads);
    int retval = elementwiseAdd2DMatrix_MT(dM, M_scaled, OUT,number_of_threads);
    destroy2DMatrix_MT(M_scaled,number_of_threads);
    return retval;
}

int momentumUpdate_MT(TwoDMatrix* X, TwoDMatrix* dX, TwoDMatrix* v, float mu, float learning_rate,  TwoDMatrix* OUT, int number_of_threads) {
    init2DMatrix_MT(OUT, X->height, X->width, number_of_threads);
    TwoDMatrix* v_mu = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(v_mu, v->height, v->width, number_of_threads);
    elementMul_MT(v, mu, v_mu, number_of_threads);
    TwoDMatrix* dX_scaled = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(dX_scaled, X->height, X->width, number_of_threads);
    elementMul_MT(dX,learning_rate,dX_scaled, number_of_threads);
    int retval = elementwiseSub2DMatrix_MT(v_mu,dX_scaled,v, number_of_threads);
    destroy2DMatrix_MT(v_mu, number_of_threads);
    destroy2DMatrix_MT(dX_scaled, number_of_threads);
    retval += elementwiseAdd2DMatrix_MT(X,v,OUT, number_of_threads);
    return retval;
}

int NAGUpdate_MT(TwoDMatrix* X, TwoDMatrix* dX, TwoDMatrix* v, TwoDMatrix* v_prev, float mu, float learning_rate,  TwoDMatrix* OUT, int number_of_threads) {
    init2DMatrix_MT(OUT, X->height, X->width, number_of_threads);
    copyTwoDMatrix(v,v_prev);
    TwoDMatrix* v_mu = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(v_mu, v->height, v->width, number_of_threads);
    elementMul_MT(v, mu, v_mu, number_of_threads);
    TwoDMatrix* dX_scaled = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(dX_scaled, X->height, X->width, number_of_threads);
    elementMul_MT(dX,learning_rate,dX_scaled, number_of_threads);
    elementwiseSub2DMatrix_MT(v_mu,dX_scaled,v, number_of_threads);
    destroy2DMatrix_MT(v_mu, number_of_threads);
    destroy2DMatrix_MT(dX_scaled, number_of_threads);
    TwoDMatrix* v_prev_mu = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(v_prev_mu, v->height, v->width, number_of_threads);
    elementMul_MT(v_prev, -1*mu, v_prev_mu, number_of_threads);
    TwoDMatrix* v_mu_plus_one = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(v_mu_plus_one, v->height, v->width, number_of_threads);
    elementMul_MT(v, 1+mu, v_mu_plus_one, number_of_threads);
    TwoDMatrix* X_update = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(X_update, v->height, v->width, number_of_threads);
    elementwiseAdd2DMatrix_MT(X,X_update,OUT, number_of_threads);
    destroy2DMatrix_MT(v_prev_mu, number_of_threads);
    destroy2DMatrix_MT(v_mu_plus_one, number_of_threads);
    destroy2DMatrix_MT(X_update, number_of_threads);
    return 0;
}

int RMSProp_MT(TwoDMatrix* X, TwoDMatrix* dX, TwoDMatrix* cache, float learning_rate, float decay_rate, float eps, TwoDMatrix* OUT, int number_of_threads) {
    TwoDMatrix* cache_scaled = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(cache_scaled, X->height, X->width, number_of_threads);
    elementMul_MT(cache,decay_rate,cache_scaled, number_of_threads);
    TwoDMatrix* dX_squared = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(dX_squared, dX->height, dX->width, number_of_threads);
    elementwiseMul2DMatrix_MT(dX,dX,dX_squared, number_of_threads);
    TwoDMatrix* dX_squared_scaled = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(dX_squared_scaled, dX->height, dX->width, number_of_threads);
    elementMul_MT(dX_squared,1-decay_rate,dX_squared_scaled, number_of_threads);
    elementwiseAdd2DMatrix_MT(cache_scaled,dX_squared_scaled,cache, number_of_threads);
    destroy2DMatrix_MT(cache_scaled, number_of_threads);
    destroy2DMatrix_MT(dX_squared, number_of_threads);
    destroy2DMatrix_MT(dX_squared_scaled, number_of_threads);
    TwoDMatrix* cache_sqrt = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(cache_sqrt, X->height, X->width, number_of_threads);
    elementSqrt_MT(cache,cache_sqrt, number_of_threads);
    TwoDMatrix* cache_sqrt_eps = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(cache_sqrt_eps, X->height, X->width, number_of_threads);
    elementAdd_MT(cache_sqrt, eps, cache_sqrt_eps, number_of_threads);
    TwoDMatrix* dX_scaled = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(dX_scaled, X->height, X->width, number_of_threads);
    elementMul_MT(dX,learning_rate,dX_scaled, number_of_threads);
    TwoDMatrix* X_update = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(X_update, dX->height, dX->width, number_of_threads);
    elementwiseDiv2DMatrix_MT(dX_scaled,cache_sqrt_eps,X_update, number_of_threads);
    elementwiseSub2DMatrix_MT(X,X_update,OUT, number_of_threads);
    destroy2DMatrix_MT(cache_sqrt, number_of_threads);
    destroy2DMatrix_MT(cache_sqrt_eps, number_of_threads);
    destroy2DMatrix_MT(dX_scaled, number_of_threads);
    destroy2DMatrix_MT(X_update, number_of_threads);
    return 0;
}

int batchnorm_training_forward_MT(TwoDMatrix* M, float momentum, float eps, TwoDMatrix* gamma, TwoDMatrix* beta, TwoDMatrix* OUT, TwoDMatrix* mean_cache, TwoDMatrix* var_cache, TwoDMatrix* mean, TwoDMatrix* var, TwoDMatrix* M_normalized, int number_of_threads) {
    matrixYMeanVar(M, mean, var);
    TwoDMatrix* M_centered = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(M_centered,M->height,M->width, number_of_threads);
    broadcastSub_MT(M,mean,1,M_centered, number_of_threads);
    TwoDMatrix* var_eps = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(var_eps,var->height,var->width, number_of_threads);
    TwoDMatrix* std_var = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(std_var,var->height,var->width, number_of_threads);
    elementAdd_MT(var,eps,var_eps, number_of_threads);
    elementSqrt_MT(var_eps,std_var, number_of_threads);
    init2DMatrix_MT(M_normalized,M->height,M->width, number_of_threads);
    broadcastDiv_MT(M_centered,std_var,1,M_normalized, number_of_threads);
    init2DMatrix_MT(OUT, M->height, M->width, number_of_threads);
    broadcastMul_MT(M_normalized,gamma,1,M_normalized, number_of_threads);
    broadcastAdd_MT(M_normalized,beta,1,OUT, number_of_threads);
    TwoDMatrix* mean_scaled = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(mean_scaled,mean->height,mean->width, number_of_threads);
    TwoDMatrix* var_scaled = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(var_scaled,var->height,var->width, number_of_threads);
    elementMul_MT(mean_cache,momentum,mean_cache, number_of_threads);
    elementMul_MT(mean,1-momentum,mean_scaled, number_of_threads);
    elementwiseAdd2DMatrix_MT(mean_cache,mean_scaled,mean_cache, number_of_threads);
    elementMul_MT(var_cache,momentum,var_cache, number_of_threads);
    elementMul_MT(var,1-momentum,var_scaled, number_of_threads);
    elementwiseAdd2DMatrix_MT(var_cache,var_scaled,var_cache, number_of_threads);
    destroy2DMatrix_MT(M_centered, number_of_threads);
    destroy2DMatrix_MT(var_eps, number_of_threads);
    destroy2DMatrix_MT(std_var, number_of_threads);
    destroy2DMatrix_MT(mean_scaled, number_of_threads);
    destroy2DMatrix_MT(var_scaled, number_of_threads);
    return 0;
}

int batchnorm_test_forward_MT(TwoDMatrix* M, TwoDMatrix* mean_cache, TwoDMatrix* var_cache, float eps, TwoDMatrix* gamma, TwoDMatrix* beta, TwoDMatrix* OUT, int number_of_threads) {
    TwoDMatrix* M_centered = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(M_centered,M->height,M->width, number_of_threads);
    broadcastSub_MT(M,mean_cache,1,M_centered, number_of_threads);
    TwoDMatrix* var_eps = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(var_eps,var_cache->height,var_cache->width, number_of_threads);
    TwoDMatrix* std_var = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(std_var,var_cache->height,var_cache->width, number_of_threads);
    elementAdd_MT(var_cache,eps,var_eps, number_of_threads);
    elementSqrt_MT(var_eps,std_var, number_of_threads);
    TwoDMatrix* M_normalized = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(M_normalized,M->height,M->width, number_of_threads);
    broadcastDiv_MT(M_centered,std_var,1,M_normalized, number_of_threads);
    init2DMatrix_MT(OUT, M->height, M->width, number_of_threads);
    broadcastMul_MT(M_normalized,gamma,1,M_normalized, number_of_threads);
    broadcastAdd_MT(M_normalized,beta,1,OUT, number_of_threads);
    destroy2DMatrix_MT(M_centered, number_of_threads);
    destroy2DMatrix_MT(var_eps, number_of_threads);
    destroy2DMatrix_MT(std_var, number_of_threads);
    destroy2DMatrix_MT(M_normalized, number_of_threads);
    return 0;
}

int batchnorm_backward_MT(TwoDMatrix* dOUT, TwoDMatrix* M, TwoDMatrix* M_normalized, TwoDMatrix* gamma, TwoDMatrix* beta, TwoDMatrix* mean, TwoDMatrix* var, float eps, TwoDMatrix* dM, TwoDMatrix* dgamma, TwoDMatrix* dbeta, int number_of_threads) {
    init2DMatrix_MT(dbeta,beta->height,beta->width, number_of_threads);
    sumY2DMatrix(dOUT,dbeta);
    init2DMatrix_MT(dgamma,gamma->height,gamma->width, number_of_threads);
    TwoDMatrix* dOUT_M_normalized = matrixMalloc(sizeof(TwoDMatrix));
    elementwiseMul2DMatrix_MT(dOUT, M_normalized,dOUT_M_normalized, number_of_threads);
    sumY2DMatrix(dOUT_M_normalized,dgamma);
    TwoDMatrix* dM_normalized = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(dM_normalized, M_normalized->height, M_normalized->width, number_of_threads);
    broadcastMul_MT(dOUT,gamma,1,dM_normalized, number_of_threads);
    TwoDMatrix* M_mu = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(M_mu, M_normalized->height, M_normalized->width, number_of_threads);
    broadcastSub_MT(M,mean,1,M_mu, number_of_threads);
    TwoDMatrix* std_var = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(std_var, var->height, var->width, number_of_threads);
    elementAdd_MT(var, eps, std_var, number_of_threads);
    elementSqrt_MT(std_var,std_var, number_of_threads);
    TwoDMatrix* std_var_inv = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(std_var_inv, var->height, var->width, number_of_threads);
    for(int i=0;i<std_var_inv->height;i++) {
        for(int j=0;j<std_var_inv->width;j++) {
            std_var_inv->d[i][j] = 1.0/std_var->d[i][j];
        }
    }
    TwoDMatrix* std_var_inv_cube = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(std_var_inv_cube, var->height, var->width, number_of_threads);
    elementwiseMul2DMatrix_MT(std_var_inv,std_var_inv,std_var_inv_cube, number_of_threads);
    elementwiseMul2DMatrix_MT(std_var_inv_cube,std_var_inv,std_var_inv_cube, number_of_threads);
    TwoDMatrix* M_tmp = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(M_tmp,M->height,M->width, number_of_threads);
    elementwiseMul2DMatrix_MT(dM_normalized,M_mu,M_tmp, number_of_threads);
    TwoDMatrix* dvar = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(dvar, var->height, var->width, number_of_threads);
    sumY2DMatrix(M_tmp,dvar);
    elementMul_MT(dvar,-0.5,dvar, number_of_threads);
    elementwiseMul2DMatrix_MT(dvar,std_var_inv_cube,dvar, number_of_threads);
    TwoDMatrix* M_mu_mean = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(M_mu_mean,mean->height,mean->width, number_of_threads);
    matrixYMeanVar(M_mu, M_mu_mean, NULL);
    TwoDMatrix* var_tmp = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(var_tmp,var->height,var->width, number_of_threads);
    elementwiseMul2DMatrix_MT(dvar,M_mu_mean,var_tmp, number_of_threads);
    elementMul_MT(var_tmp,2.0,var_tmp, number_of_threads);
    TwoDMatrix* dmean = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(dmean, mean->height, mean->width, number_of_threads);
    TwoDMatrix* M_tmp2 = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(M_tmp2,M->height,M->width, number_of_threads);
    broadcastMul_MT(dM_normalized,std_var_inv,1,M_tmp2, number_of_threads);
    TwoDMatrix* mean_tmp = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(mean_tmp,mean->height,mean->width, number_of_threads);
    sumY2DMatrix(M_tmp2,mean_tmp);
    elementMul_MT(M_tmp2,-1.0,M_tmp2, number_of_threads);
    elementwiseSub2DMatrix_MT(M_tmp2,var_tmp,dmean, number_of_threads);
    TwoDMatrix* dM1 = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(dM1,M->height,M->width, number_of_threads);
    broadcastMul_MT(dM_normalized,std_var_inv,1,dM1, number_of_threads);
    TwoDMatrix* dM2 = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(dM2,M->height,M->width, number_of_threads);
    broadcastMul_MT(M_mu,dvar,1,dM2, number_of_threads);
    elementMul_MT(dM2,2.0,dM2, number_of_threads);
    elementDiv_MT(dM2,dM2->height,dM2, number_of_threads);
    elementwiseAdd2DMatrix_MT(dM1,dM2,dM, number_of_threads);
    TwoDMatrix* dmean_tmp = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(dmean_tmp,dmean->height,dmean->width, number_of_threads);
    elementDiv_MT(dmean,M->height,dmean_tmp, number_of_threads);
    broadcastAdd_MT(dM,dmean_tmp,1,dM, number_of_threads);
    destroy2DMatrix_MT(dM_normalized, number_of_threads);
    destroy2DMatrix_MT(M_mu, number_of_threads);
    destroy2DMatrix_MT(std_var, number_of_threads);
    destroy2DMatrix_MT(std_var_inv, number_of_threads);
    destroy2DMatrix_MT(std_var_inv_cube, number_of_threads);
    destroy2DMatrix_MT(M_tmp, number_of_threads);
    destroy2DMatrix_MT(dvar, number_of_threads);
    destroy2DMatrix_MT(M_mu_mean, number_of_threads);
    destroy2DMatrix_MT(var_tmp, number_of_threads);
    destroy2DMatrix_MT(dmean, number_of_threads);
    destroy2DMatrix_MT(M_tmp2, number_of_threads);
    destroy2DMatrix_MT(mean_tmp, number_of_threads);
    destroy2DMatrix_MT(dM1, number_of_threads);
    destroy2DMatrix_MT(dM2, number_of_threads);
    destroy2DMatrix_MT(dmean_tmp, number_of_threads);
    destroy2DMatrix_MT(dOUT_M_normalized, number_of_threads);
    return 0;
}
