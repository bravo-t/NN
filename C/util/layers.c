#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include "matrix_type.h"
#include "misc_utils.h"
#include "matrix_operations.h"

int affineLayerForward(TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* b, TwoDMatrix* OUT) {
    init2DMatrix(OUT, X->height, W->width);
    if (dotProduct(X,W,OUT)) {
        printf("ERROR: Input matrix size mismatch: X->width = %d, W->height = %d\n", X->width,W->height);
        exit(1);
    }
    broadcastAdd(OUT, b, 1, OUT);
    return 0;
}

int affineLayerBackword(TwoDMatrix* dOUT, TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* b, TwoDMatrix* dX, TwoDMatrix* dW, TwoDMatrix* db) {
    init2DMatrix(dX, X->height, X->width);
    init2DMatrix(dW, W->height, W->width);
    init2DMatrix(db, b->height, b->width);
    TwoDMatrix* XT = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* WT = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(XT, X->width, X->height);
    init2DMatrix(WT, W->width, W->height);
    transpose2DMatrix(X, XT);
    transpose2DMatrix(W, WT);
    if (dotProduct(dOUT,WT,dX)) {
        printf("ERROR: Input matrix size mismatch: dOUT->width = %d, W.T->height = %d\n", dOUT->width,WT->height);
        exit(1);
    }
    if (dotProduct(XT,dOUT,dW)) {
        printf("ERROR: Input matrix size mismatch: X.T->width = %d, dOUT->height = %d\n", XT->width,dOUT->height);
        exit(1);
    }
    if (db->height == 1) {
        sumY2DMatrix(dOUT,db);
    } else {
        sumX2DMatrix(dOUT,db);
    }
    destroy2DMatrix(XT);
    destroy2DMatrix(WT);
    return 0;
}

int leakyReLUForward(TwoDMatrix* M, float alpha, TwoDMatrix* OUT) {
    return elementLeakyReLU(M, alpha, OUT);
}

int vanillaUpdate(TwoDMatrix* M, TwoDMatrix* dM, float learning_rate, TwoDMatrix* OUT) {
    init2DMatrix(OUT, M->height, M->width);
    TwoDMatrix* dM_scaled = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(dM_scaled, M->height, M->width);
    elementMul(dM,learning_rate,dM_scaled);
    int retval = elementwiseSub2DMatrix(M, dM_scaled, OUT);
    destroy2DMatrix(dM_scaled);
    return retval;
}

int leakyReLUBackward(TwoDMatrix* dM, TwoDMatrix* M, float alpha, TwoDMatrix* OUT) {
    init2DMatrix(OUT,dM->height,dM->width);

    for(int i=0;i<dM->height;i++) {
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

/*
score is a N*M 2D matrix, N is the height, and M is the width. N is the number of examples for 
a mini-batch, and M is the number of labels. The layout of score is like the following;

            score for label1    score for label2    score for label3    ...     score for labelM
         -----------------------------------------------------------------------------------------
Example1 |              ****                ****                ****    ...                 ****
Example2 |              ****                ****                ****    ...                 ****
    .    |                                            .
    .    |                                            .
    .    |                                            .
ExampleN |              ****                ****                ****    ...                 ****

correct_label is a 2D matrix with the height of N, and width of 1. Layout as follows:

            label for the correct class
         ------------------------------
Example1 |                         [0,M)
Example2 |                         [0,M)
   .     |                           .
   .     |                           .
   .     |                           .
ExampleN |                         [0,M)

 *
 * Below is an example with real digits, there's only 1 training sample in the mini-batch for simplicity:
 * score = [[-1, 5, 4, 7, 3, 2]]
 * correct_label = [[2]], means the 3rd score in score is the correct one
 * margins = [[0, 2, 0, 4, 0, 0]], after the max(0, wrong - correct + 1) operation
 * number of positive ones in margins number_of_pos = 2 
 * dscore = [[0, 1, 0, 1, 0, 0]], element is 1 if margins > 0
 * then
 * dscore = [[0, 1, 0 - 2, 1, 0, 0]] = [[0, 1, -2, 1, 0, 0]] 
 * this can be expressed as the willing to reduce the results of the 2nd and 4th score, 
 * further increase the 3rd score, which is the correct one, while leaving others unhurt, 
 * because they are smaller than the delta
 */
float SVMLoss(TwoDMatrix* score, TwoDMatrix* correct_label, TwoDMatrix* dscore) {
    TwoDMatrix* margins = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(margins, score->height, score->width);
    init2DMatrix(dscore, score->height, score->width);
    int number_of_examples = score->height;
    int* number_of_pos = calloc(number_of_examples, sizeof(int));
    // Matrix margins contains the values of score undergone the process of max(0, wrong - correct + 1) operation in hinge loss
    for(int i=0;i<score->height;i++) {
        int correct_index = correct_label->d[i][0];
        float correct_score = score->d[i][correct_index];
        for(int j=0;j!=correct_index&&j<score->width;j++) {
            margins->d[i][j] = fmaxf(0,score->d[i][j] - correct_score + 1);
            if (margins->d[i][j] > 0) {
                number_of_pos[i]++;
                /*
                 *  Why can't I just use "dscore->d[i][j] = margins->d[i][j]"?
                 *  Because this seems to be decreasing the larger wrong scores more strongly
                 */
                dscore->d[i][j] = 1;
            } else {
                dscore->d[i][j] = 0;
            }
        }
        margins->d[i][correct_index] = 0;
    }
    float data_loss = sumAll(margins) / number_of_examples;
    for(int i=0;i<score->height;i++) {
        int correct_index = correct_label->d[i][0];
        dscore->d[i][correct_index] -= number_of_pos[i];
    }
    elementDiv(dscore,number_of_examples,dscore);
    destroy2DMatrix(margins);
    free(number_of_pos);
    return data_loss;
}

float softmaxLoss(TwoDMatrix* score, TwoDMatrix* correct_label, TwoDMatrix* dscore) {
    init2DMatrix(dscore,score->height,score->width);
    TwoDMatrix* max_scores = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(max_scores,score->height,1);
    maxX2DMatrix(score,max_scores);
    //printf("score = \n");
    //printMatrix(score);
    //printf("max_scores = \n");
    //printMatrix(max_scores);
    TwoDMatrix* shifted = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(shifted,score->height,score->width);
    broadcastSub(score,max_scores,0,shifted);
    //printf("shifted = \n");
    //printMatrix(shifted);
    TwoDMatrix* exp_score = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(exp_score,score->height,score->width);
    elementExp(shifted,exp_score);
    //printf("exp_score = \n");
    //printMatrix(exp_score);
    TwoDMatrix* exp_sum = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(exp_sum,score->height,1);
    sumX2DMatrix(exp_score,exp_sum);
    //printf("exp_sum = \n");
    //printMatrix(exp_sum);
    TwoDMatrix* probs = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(probs,score->height,score->width);
    broadcastDiv(exp_score,exp_sum,0,probs);
    //printf("probs = \n");
    //printMatrix(probs);
    TwoDMatrix* correct_probs = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(correct_probs,score->height,1);
    for(int i=0;i<score->height;i++) {
        int correct_index = correct_label->d[i][0];
        for(int j=0;j<score->width;j++) {
            dscore->d[i][j] = probs->d[i][j];
            if(j == correct_index) {
                dscore->d[i][j] -= 1;
            }
        }
        correct_probs->d[i][0] = -log(probs->d[i][correct_index]);
    }
    //printf("correct_probs = \n");
    //printMatrix(correct_probs);
    int number_of_examples = score->height;
    float data_loss = sumAll(correct_probs) / number_of_examples;
    destroy2DMatrix(max_scores);
    destroy2DMatrix(shifted);
    destroy2DMatrix(exp_score);
    destroy2DMatrix(exp_sum);
    destroy2DMatrix(probs);
    destroy2DMatrix(correct_probs);
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
    elementMul(dX,-1*learning_rate,dX_scaled);
    TwoDMatrix* X_update = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(X_update, dX->height, dX->width);
    elementwiseAdd2DMatrix(X,X_update,OUT);
    destroy2DMatrix(cache_sqrt);
    destroy2DMatrix(cache_sqrt_eps);
    destroy2DMatrix(dX_scaled);
    destroy2DMatrix(X_update);
    return 0;
}

