#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
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
    return 0;
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
    maxX2DMatrix_MT(score,max_scores,number_of_threads);
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
    sumX2DMatrix_MT(exp_score,exp_sum,number_of_threads);
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

int L2RegLossBackward_MT(TwoDMatrix* dM, TwoDMatrix* M, float reg_strength, TwoDMatrix* OUT) {
    init2DMatrix_MT(OUT, M->height, M->width,number_of_threads);
    TwoDMatrix* M_scaled = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix_MT(M_scaled, M->height, M->width,number_of_threads);
    elementMul_MT(M,reg_strength,M_scaled,number_of_threads);
    int retval = elementwiseAdd2DMatrix_MT(dM, M_scaled, OUT,number_of_threads);
    destroy2DMatrix_MT(M_scaled,number_of_threads);
    return retval;
}