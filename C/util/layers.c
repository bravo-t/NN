#include "matrix_operations.h"
#include <stdlib.h>
#include <malloc.h>
#include <math.h>

int affineLayerForward(TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* b, TwoDMatrix* OUT) {
    init2DMatrix(OUT, X->height, W->width);
    if (dotProduct(X,W,OUT)) {
        printf("ERROR: Input matrix size mismatch: X->width = %d, W->height = %d\n", X->width,W->height);
        exit 1;
    }
    broadcastAdd(OUT, b, 0, OUT);
    return 0;
}

int affineLayerBackword(TwoDMatrix* dOUT, TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* b, TwoDMatrix* dX, TwoDMatrix* dW, TwoDMatrix* db) {
    init2DMatrix(dX, X->height, X->width);
    init2DMatrix(dW, W->height, W->width);
    init2DMatrix(db, b->height, b->width);
    TwoDMatrix* XT = malloc(sizeof(TwoDMatrix));
    TwoDMatrix* WT = malloc(sizeof(TwoDMatrix));
    init2DMatrix(XT, X->width, X->height);
    init2DMatrix(WT, W->width, W->height);
    if (dotProduct(dOUT,WT,dX)) {
        printf("ERROR: Input matrix size mismatch: dOUT->width = %d, W.T->height = %d\n", dOUT->width,WT->height);
        exit 1;
    }
    if (dotProduct(XT,dOUT,dW)) {
        printf("ERROR: Input matrix size mismatch: X.T->width = %d, dOUT->height = %d\n", XT->width,dOUT->height);
        exit 1;
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
    return elementLeakyReLU(TwoDMatrix* M,float alpha, TwoDMatrix* OUT);
}

int vanillaUpdate(TwoDMatrix* M, TwoDMatrix* dM, float learning_rate, TwoDMatrix* OUT) {
    init2DMatrix(OUT, M->height, M->width);
    TwoDMatrix* dM_scaled = malloc(sizeof(TwoDMatrix));
    init2DMatrix(dM_scaled, M->height, M->width);
    elementMul(dM,learning_rate,dM_scaled);
    return elementwiseAdd2DMatrix(M, dM_scaled, OUT);
}

int leakyReLUBackward(TwoDMatrix* dM, TwoDMatrix* M, float alpha, TwoDMatrix* OUT) {
    init2DMatrix(OUT,dM->height,dM->width);
    for(int i=0;i<dM->height;i++) {
        for(int j=0;j<dM->width;j++) {
            if (M->d[i][j] >= 0) {
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
    TwoDMatrix* margins = malloc(sizeof(TwoDMatrix));
    init2DMatrix(margins, score->height, score->width);
    init2DMatrix(dscore, score->height, score->width);
    int number_of_examples = score->height;
    int number_of_pos[number_of_examples] = {0};
    // Matrix margins contains the values of score undergone the process of max(0, wrong - correct + 1) operation in hinge loss
    for(int i=0;i<score->height;i++) {
        int correct_index = correct_label->d[i][0];
        float correct_score = score->d[i][correct_index];
        for(int j=0;j!=correct_index&&j<score->width;j++) {
            margins->d[i][j] = max(0,score->d[i][j] - correct_score + 1);
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
    elementDiv(dscore,number_of_examples);
    destroy2DMatrix(margins);
    return data_loss;
}

float softmaxLoss(TwoDMatrix* score, TwoDMatrix* correct_label, TwoDMatrix* dscore) {
    init2DMatrix(dscore,score->height,score->width);
    TwoDMatrix* max_scores = malloc(sizeof(TwoDMatrix));
    init2DMatrix(max_scores,score->height,1);
    maxX2DMatrix(score,max_scores);
    TwoDMatrix* shifted = malloc(sizeof(TwoDMatrix));
    init2DMatrix(shifted,score->height,score->width);
    broadcastSub(score,max_scores,0,shifted);
    TwoDMatrix* exp_score = malloc(sizeof(TwoDMatrix));
    init2DMatrix(exp_score,score->height,score->width);
    elementExp(shifted,exp_score);
    TwoDMatrix* exp_sum = malloc(sizeof(TwoDMatrix));
    init2DMatrix(exp_sum,score->height,1);
    sumX2DMatrix(exp_score,exp_sum);
    TwoDMatrix* probs = malloc(sizeof(TwoDMatrix));
    init2DMatrix(probs,score->height,score->width);
    broadcastDiv(exp_score,exp_sum,0,probs);
    TwoDMatrix* correct_probs = malloc(sizeof(TwoDMatrix));
    init2DMatrix(correct_probs,score->height,1);
    for(int i=0;i<score->height;i++) {
        int correct_index = correct_label->d[i][0];
        for(int j=0;j<score->width;j++) {
            dscore->d[i][j] = probs->d[i][j];
            if(j == correct_index) {
                dscore->d[i][j] -= 1;
            }
        }
        correct_probs->d[i][0] = -log(probs->[i][correct_index]);
    }
    int number_of_examples = socre->height;
    float data_loss = sumAll(correct_probs) / number_of_examples;
    destroy2DMatrix(max_scores);
    destroy2DMatrix(shifted);
    destroy2DMatrix(exp_score);
    destroy2DMatrix(exp_sum);
    destroy2DMatrix(probs);
    destroy2DMatrix(correct_probs);
    return data_loss;
}

float L2RegLoss(TwoDMatrix** Ms,int number_of_weights, float reg_strength) {
    float reg_loss = 0;
    for (int i = 0; i < number_of_weights; i++) {
        TwoDMatrix* M_squared = malloc(sizeof(TwoDMatrix));
        init2DMatrix(M_squared,Ms[i]->height,Ms[i]->width);
        elementwiseMul2DMatrix(Ms[i],Ms[i],M_squared);
        reg_loss += 0.5*reg_strength*sumAll(M_squared);
        destroy2DMatrix(M_squared);
    }
    return reg_loss;
}
