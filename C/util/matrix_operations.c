#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <stdbool.h>
#include "matrix_type.h"
#include "misc_utils.h"
#include "matrix_operations.h"

float frand() {
    return (rand()+1.0)/(RAND_MAX+1.0);
}

float random_normal(float mean, float std) {
    return mean + std*(sqrt(-2*log(frand()))*cos(2*M_PI*frand()));
}

int init2DMatrix(TwoDMatrix* M, int height, int width) {
    if (M->initialized) return 0;
    M->height = height;
    M->width = width;
    float** data = (float**) calloc(height, sizeof(float*));
    for(int i = 0; i<height;i++) {
        data[i] = (float*) calloc(width,sizeof(float));
    }
    M->d = data;
    M->initialized = true;
    return 0;
}

int init2DMatrixNormRand(TwoDMatrix* M, int height, int width, float mean, float std) {
    M->height = height;
    M->width = width;
    float** data = (float**) calloc(height,sizeof(float*));
    for(int i = 0; i<height;i++) {
        data[i] = (float*) calloc(width,sizeof(float));
        for(int j=0;j<width;j++) {
            data[i][j] = random_normal(mean,std);
        }
    }
    M->d = data;
    return 0;
}

int init2DMatrixZero(TwoDMatrix* M, int height, int width) {
    M->height = height;
    M->width = width;
    float** data = (float**) calloc(height,sizeof(float*));
    for(int i = 0; i<height;i++) {
        data[i] = (float*) calloc(width,sizeof(float));
        for(int j=0;j<width;j++) {
            data[i][j] = 0;
        }
    }
    M->d = data;
    return 0;
}

int init2DMatrixOne(TwoDMatrix* M, int height, int width) {
    M->height = height;
    M->width = width;
    float** data = (float**) calloc(height,sizeof(float*));
    for(int i = 0; i<height;i++) {
        data[i] = (float*) calloc(width,sizeof(float));
        for(int j=0;j<width;j++) {
            data[i][j] = 1;
        }
    }
    M->d = data;
    return 0;
}

int copyTwoDMatrix(TwoDMatrix* M, TwoDMatrix* OUT) {
    int retval = init2DMatrix(OUT, M->height, M->width);
    for(int i=0;i<M->height;i++) {
        for(int j=0;j<M->width;j++) {
            OUT->d[i][j] = M->d[i][j];
        }
    }
    return retval;
}

int destroy2DMatrix(TwoDMatrix* M) {
    for(int i=0;i<M->height;i++) {
        free(M->d[i]);
        M->d[i] = NULL;
    }
    free(M->d);
    M->d = NULL;
    free(M);
    M = NULL;
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
        for(int j=0;j<M->width;j++) OUT->d[i][0] += M->d[i][j];
    }
    return 0;
}

int maxX2DMatrix(TwoDMatrix* M,TwoDMatrix* OUT) {
    init2DMatrix(OUT, M->height,1);
    for(int i=0;i<M->height;i++) {
        OUT->d[i][0] = M->d[i][0];
        for(int j=0;j<M->width;j++) OUT->d[i][0] = fmaxf(OUT->d[i][0], M->d[i][j]);
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
        for(int j=0;j<M->height;j++) OUT->d[0][i] += M->d[j][i];
    }
    return 0;
}

int maxY2DMatrix(TwoDMatrix* M,TwoDMatrix* OUT) {
    init2DMatrix(OUT, 1,M->width);
    for(int i=0;i<M->width;i++) {
        OUT->d[0][i] = M->d[0][i];
        for(int j=0;j<M->height;j++) OUT->d[0][i] = fmaxf(OUT->d[0][i], M->d[j][i]);
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

int elementwiseSub2DMatrix(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT) {
    if (A->height != B->height) return 1;
    if (A->width != B->width) return 1;
    init2DMatrix(OUT,A->height,A->width);
    for(int i=0;i<A->height;i++) {
        for(int j=0;j<A->width;j++) {
            OUT->d[i][j] = A->d[i][j] - B->d[i][j];
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

int elementwiseDiv2DMatrix(TwoDMatrix* A, TwoDMatrix* B, TwoDMatrix* OUT) {
    if (A->height != B->height) return 1;
    if (A->width != B->width) return 1;
    init2DMatrix(OUT,A->height,A->width);
    for(int i=0;i<A->height;i++) {
        for(int j=0;j<A->width;j++) {
            OUT->d[i][j] = A->d[i][j] / B->d[i][j];
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
    return elementMul(M, n, OUT);
}

int elementSqrt(TwoDMatrix* M, TwoDMatrix* OUT) {
    init2DMatrix(OUT,M->height,M->width);
    for(int i=0;i<M->height;i++) {
        for(int j=0;j<M->width;j++) {
            OUT->d[i][j] = sqrt(M->d[i][j]);
        }
    }
    return 0;
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
    TwoDMatrix *broadcasted = matrixMalloc(sizeof(TwoDMatrix));
    int n;
    if (direction == 0) {
        n = M->width;
    } else {
        n = M->height;
    }
    if (broadcastMatrix(b,n,direction,broadcasted)) {
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

int broadcastSub(TwoDMatrix* M, TwoDMatrix* b, int direction, TwoDMatrix* OUT) {
    TwoDMatrix *broadcasted = matrixMalloc(sizeof(TwoDMatrix));
    int n;
    if (direction == 0) {
        n = M->width;
    } else {
        n = M->height;
    }
    if (broadcastMatrix(b,n,direction,broadcasted)) {
        destroy2DMatrix(broadcasted);
        return 1;
    }
    if (elementwiseSub2DMatrix(M,broadcasted,OUT)) {
        destroy2DMatrix(broadcasted);
        return 1;
    }
    destroy2DMatrix(broadcasted);
    return 0;
}

int broadcastMul(TwoDMatrix* M, TwoDMatrix* b, int direction, TwoDMatrix* OUT) {
    TwoDMatrix *broadcasted = matrixMalloc(sizeof(TwoDMatrix));
    int n;
    if (direction == 0) {
        n = M->width;
    } else {
        n = M->height;
    }
    if (broadcastMatrix(b,n,direction,broadcasted)) {
        destroy2DMatrix(broadcasted);
        return 1;
    }
    if (elementwiseMul2DMatrix(M,broadcasted,OUT)) {
        destroy2DMatrix(broadcasted);
        return 1;
    }
    destroy2DMatrix(broadcasted);
    return 0;
}

int broadcastDiv(TwoDMatrix* M, TwoDMatrix* b, int direction, TwoDMatrix* OUT) {
    TwoDMatrix *broadcasted = matrixMalloc(sizeof(TwoDMatrix));
    int n;
    if (direction == 0) {
        n = M->width;
    } else {
        n = M->height;
    }
    if (broadcastMatrix(b,n,direction,broadcasted)) {
        destroy2DMatrix(broadcasted);
        return 1;
    }
    if (elementwiseDiv2DMatrix(M,broadcasted,OUT)) {
        destroy2DMatrix(broadcasted);
        return 1;
    }
    destroy2DMatrix(broadcasted);
    return 0;
}

int chop2DMatrix(TwoDMatrix* M, int height_start, int height_end, TwoDMatrix* OUT) {
    if (height_start >= M->height || height_end >= M->height) {
        printf("ERROR: Out of boundary in chop2DMatrix. Requesting %d - %d, but max index of the matrix is %d\n",height_start,height_end, M->height-1);
        return 1;
    }
    init2DMatrix(OUT,height_end-height_start+1,M->width);
    for(int i=height_start;i<=height_end;i++) {
        int index = i - height_start;
        for(int j=0;j<M->width;j++) {
            OUT->d[index][j] = M->d[i][j];
        }
    }
    return 0;
}

int matrixYMeanVar(TwoDMatrix* M, TwoDMatrix* mean, TwoDMatrix* var) {
    init2DMatrix(mean, 1, M->width);
    if (var != NULL) {
        init2DMatrix(var, 1, M->width);
    }
    for(int i=0;i<M->width;i++) {
        float sum = 0;
        for(int j=0;j<M->height;j++) {
            sum += M->d[j][i];
        }
        mean->d[0][i] = sum/M->height;
        if (var != NULL) {
            float variance = 0;
            for(int j=0;j<M->height;j++) {
                variance += (M->d[j][i] - mean->d[0][i])*(M->d[j][i] - mean->d[0][i]);
            }
            var->d[0][i] = variance/M->height;
        }
    }
    return 0;
}

int matrixXMeanVar(TwoDMatrix* M, TwoDMatrix* mean, TwoDMatrix* var) {
    init2DMatrix(mean, M->height, 1);
    if (var != NULL) {
        init2DMatrix(var, M->height, 1);
    }
    for(int i=0;i<M->height;i++) {
        float sum = 0;
        for(int j=0;j<M->width;j++) {
            sum += M->d[i][j];
        }
        mean->d[i][0] = sum/M->width;
        if (var != NULL) {
            float variance = 0;
            for(int j=0;j<M->width;j++) {
                variance += (M->d[i][j] - mean->d[i][0])*(M->d[i][j] - mean->d[i][0]);
            }
            var->d[0][i] = variance/M->width;
        }
    }
    return 0;
}


// misc functions

TwoDMatrix* load2DMatrixFromFile(char* filename) {
    FILE* fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("ERROR: Cannot open %s to read\n",filename);
        exit(1);
    }
    char buff[8192];
    fscanf(fp,"%s",buff);
    int height,width;
    fscanf(fp,"%d",&height);
    fscanf(fp,"%d",&width);
    float value;
    TwoDMatrix* M = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(M,height,width);
    for(int i=0;i<height;i++) {
        for(int j=0;j<width;j++) {
            fscanf(fp,"%f",&value);
            M->d[i][j] = value;
        }
    }
    fclose(fp);
    return M;
}

float matrixError(TwoDMatrix* a, TwoDMatrix* b) {
    if (a->height != b->height) {
        printf("HOLY ERROR: Height does not match, your code is really messed up\n");
        return 1.0/0.0;
    }
    if (a->width != b->width) {
        printf("ANOTHER ERROR: Width doesn't match. FIX THEM\n");
        return 1.0/0.0;
    }
    TwoDMatrix* sub = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(sub, a->height, a->width);
    elementwiseSub2DMatrix(a, b, sub);
    float error = 0;
    for(int i=0;i<sub->height;i++) {
        for(int j=0;j<sub->width;j++) {
            if (sub->d[i][j] > 0) {
                error += sub->d[i][j];
            } else {
                error -= sub->d[i][j];
            }
        }
    }
    destroy2DMatrix(sub);
    return error;
}

void printMatrix(TwoDMatrix *M) {
    printf("Height of matrix: %d, width: %d\n",M->height,M->width);
    for(int i=0;i<M->height;i++) {
        for(int j=0;j<M->width;j++) {
            printf("%f\t",M->d[i][j]);
        }
        printf("\n");
    }
}

void __debugPrintMatrix(TwoDMatrix *M, char* M_name) {
    printf("%s = \n",M_name);
    printf("Height of matrix: %d, width: %d\n",M->height,M->width);
    for(int i=0;i<M->height;i++) {
        for(int j=0;j<M->width;j++) {
            printf("%f\t",M->d[i][j]);
        }
        printf("\n");
    }
}


void checkMatrixDiff(TwoDMatrix* a, TwoDMatrix* b, float thres) {
    float diff = matrixError(a, b);
    if (diff >= thres) {
        printf("ERROR: Difference between ref and impl is too big: %f\n",diff);
        printf("ref = \n");
        printMatrix(a);
        printf("impl = \n");
        printMatrix(b);
    } else {
        printf("Difference of the two matrixes are %f\n",diff);
    }
}
