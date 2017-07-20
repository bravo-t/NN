#include "matrix_operations.h"
#include <stdlib.h>
#include <malloc.h>

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

            score for label1    score for label2    score for label3    ...     score for labelN
         -----------------------------------------------------------------------------------------
Example1 |              ****                ****                ****    ...                 ****
Example2 |              ****                ****                ****    ...                 ****
         |                                            .
         |                                            .
         |                                            .
ExampleM |              ****                ****                ****    ...                 ****


 *
 */
int SVMLoss(TwoDMatrix* score, TwoDMatrix* correct_class, TwoDMatrix* dsocre, float data_loss) {
    
}
