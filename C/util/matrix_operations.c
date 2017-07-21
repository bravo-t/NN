#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include "matrix_type.h"
#include "matrix_operations.h"

float frand() {
    return (rand()+1.0)/(RAND_MAX+1.0);
}

float random_normal(float mean, float std) {
    return mean + std*(sqrt(-2*log(frand()))*cos(2*M_PI*frand()));
}

int init2DMatrix(TwoDMatrix* M, int height, int width) {
    M->height = height;
    M->width = width;
    float** data = (float**) malloc(sizeof(float*)*height);
    for(int i = 0; i<height;i++) {
        data[i] = (float*) malloc(sizeof(float)*width);
    }
    M->d = data;
    return 0;
}

int init2DMatrixNormRand(TwoDMatrix* M, int height, int width, float mean, float std) {
    M->height = height;
    M->width = width;
    float** data = (float**) malloc(sizeof(float*)*height);
    for(int i = 0; i<height;i++) {
        data[i] = (float*) malloc(sizeof(float)*width);
        for(int j=0;j<width;j++) {
            data[i][j] = random_normal(mean,std);
        }
    }
    M->d = data;
    return 0;
}

int destroy2DMatrix(TwoDMatrix* M) {
    for(int i=0;i<M->height;i++) free(M->d[i]);
    free(M);
    return 0;
}

int transpose2DMatrix(TwoDMatrix* M,TwoDMatrix* OUT) {
    init2DMatrix(OUT, M->width,M->height);
    for(int i=0;i<M->height;i++) {
        for(int j=0;j<M->width;j++) OUT->d[j][i] = M->d[i][j];
    }
    return 0;
}

int dotProduct(TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* OUT) {
    if (X->width != W->height) {
        return 1;
    }
    init2DMatrix(OUT,X->height,W->width);
    for(int i=0;i<X->height;i++) {
        for(int j=0;j<W->width;j++) {
            float sum = 0;
            for(int p=0;p<X->width;p++) sum += X->d[i][p]*W->d[p][j];
            OUT->d[i][j] = sum;
        }
    }
    return 0;
}

/*
Add all elements horizontally, result matrix will always have width = 1. For exmaple:
[[1, 0, 0],      [[1],
 [1, 1, 0],   ->  [2],
 [1, 1, 1]]       [3]]
 */
int sumX2DMatrix(TwoDMatrix* M,TwoDMatrix* OUT) {
    init2DMatrix(OUT, M->height,1);
    for(int i=0;i<M->height;i++) {
        OUT->d[i][0] = 0;
        for(int j=0;j<M->width;j++) OUT->d[i][0]+= M->d[i][j];
    }
    return 0;
}

/*
Add all elements vertically, result matrix will always have height = 1. For exmaple:
[[1, 0, 0],
 [1, 1, 0],
 [1, 1, 1]]
     |
     v
[[3, 2, 1]]
 */
int sumY2DMatrix(TwoDMatrix* M,TwoDMatrix* OUT) {
    init2DMatrix(OUT, 1,M->width);
    for(int i=0;i<M->width;i++) {
        OUT->d[0][i] = 0;
        for(int j=0;j<M->height;j++) OUT->d[0][i]+= M->d[j][i];
    }
    return 0;
}

float sumAll(TwoDMatrix* M) {
    float sum = 0;
    for(int i=0;i<M->height;i++) {
        for(int j=0;j<M->width;j++) sum += M->d[i][j];
    }
    return sum;
}

}

int elementwiseAdd2DMatrix(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT) {
    if (A->height != B->height) return 1;
    if (A->width != B->width) return 1;
    init2DMatrix(OUT,A->height,A->width);
    for(int i=0;i<A->height;i++) {
        for(int j=0;j<A->width;j++) {
            OUT->d[i][j] = A->d[i][j] + B->d[i][j];
        }
    }
    return 0;
}

int elementwiseMul2DMatrix(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT) {
    if (A->height != B->height) return 1;
    if (A->width != B->width) return 1;
    init2DMatrix(OUT,A->height,A->width);
    for(int i=0;i<A->height;i++) {
        for(int j=0;j<A->width;j++) {
            OUT->d[i][j] = A->d[i][j] * B->d[i][j];
        }
    }
    return 0;
}

int elementExp(TwoDMatrix* M,TwoDMatrix* OUT) {
    init2DMatrix(OUT,M->height,M->width);
    for(int i=0;i<M->height;i++) {
        for(int j=0;j<M->width;j++) {
            OUT->d[i][j] = exp(M->d[i][j]);
        }
    }
    return 0;
}

int elementAdd(TwoDMatrix* M, float a,TwoDMatrix* OUT) {
    init2DMatrix(OUT,M->height,M->width);
    for(int i=0;i<M->height;i++) {
        for(int j=0;j<M->width;j++) {
            OUT->d[i][j] = a+M->d[i][j];
        }
    }
    return 0;
}

int elementMul(TwoDMatrix* M, float a,TwoDMatrix* OUT) {
    init2DMatrix(OUT,M->height,M->width);
    for(int i=0;i<M->height;i++) {
        for(int j=0;j<M->width;j++) {
            OUT->d[i][j] = a*M->d[i][j];
        }
    }
    return 0;
}

int elementDiv(TwoDMatrix* M,float a, TwoDMatrix* OUT) {
    float n = 1/a;
    return elementMul(TwoDMatrix* M, float n,TwoDMatrix* OUT);
}


int elementLeakyReLU(TwoDMatrix* M,float alpha, TwoDMatrix* OUT) {
    init2DMatrix(OUT,M->height,M->width);
    for(int i=0;i<M->height;i++) {
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

int broadcastMatrix(TwoDMatrix* M, int n, int direction, TwoDMatrix* OUT) {
    if (direction == 0) {
        if (M->width != 1) {
            printf("ERROR: Cannot horizontally broadcast matrix with a width that is not 1\n");
            return 1;
        }
        init2DMatrix(OUT, M->height, n);
        for(int i=0;i<M->height;i++) {
            for(int j=0;j<n;j++) {
                OUT->d[i][j] = M->d[i][0];
            }
        }
    } else {
        if (M->height != 1) {
            printf("ERROR: Cannot vertically broadcast matrix with a height that is not 1\n");
            return 1;
        }
        init2DMatrix(OUT, n, M->width);
        for(int i=0;i<n;i++) {
            for(int j=0;j<M->width;j++) {
                OUT->d[i][j] = M->d[0][j];
            }
        }
    }
    return 0;
}

int broadcastAdd(TwoDMatrix* M, TwoDMatrix* b, int direction, TwoDMatrix* OUT) {
    TwoDMatrix *broadcasted = malloc(sizeof(TwoDMatrix));
    int n;
    if (direction == 0) {
        n = M->width;
    } else {
        n = M->height;
    }
    if (broadcastMatrix(M,n,direction,broadcasted)) {
        destroy2DMatrix(broadcasted);
        return 1;
    }
    if (elementwiseAdd2DMatrix(M,broadcasted,OUT)) {
        destroy2DMatrix(broadcasted);
        return 1;
    }
    destroy2DMatrix(broadcasted);
    return 0;
}
