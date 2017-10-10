#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <stdbool.h>
#include <string.h>
#include "network_type.h"
#include "matrix_operations.h"

float frand() {
    return (rand()+1.0)/(RAND_MAX+1.0);
}

float random_normal(float mean, float std) {
    return mean + std*(sqrt(-2*log(frand()))*cos(2*M_PI*frand()));
}

void* matrixMalloc(int size) {
    TwoDMatrix* M = malloc(size);
    memset(M,0,size);
    M->initialized = false;
    return M;
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

int init2DMatrixNormRand(TwoDMatrix* M, int height, int width, float mean, float std, int n) {
    M->height = height;
    M->width = width;
    float** data = (float**) calloc(height,sizeof(float*));
    for(int i = 0; i<height;i++) {
        data[i] = (float*) calloc(width,sizeof(float));
        for(int j=0;j<width;j++) {
            data[i][j] = random_normal(mean,std)*sqrt(2.0/n);
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

int init3DMatrix(ThreeDMatrix* M, int depth, int height, int width) {
    if (M->initialized) return 0;
    M->height = height;
    M->width = width;
    M->depth = depth;
    float*** data = (float***) calloc(depth, sizeof(float**));
    for(int i = 0; i<depth;i++) {
        data[i] = (float**) calloc(height,sizeof(float*));
        for(int j=0;j<height;j++) data[i][j] = (float*) calloc(width,sizeof(float));
    }
    M->d = data;
    M->initialized = true;
    return 0;
}

int init3DMatrixNormRand(ThreeDMatrix* M, int depth, int height, int width, float mean, float std, int n) {
    if (M->initialized) return 0;
    M->height = height;
    M->width = width;
    M->depth = depth;
    float*** data = (float***) calloc(depth, sizeof(float**));
    for(int i = 0; i<depth;i++) {
        data[i] = (float**) calloc(height,sizeof(float*));
        for(int j=0;j<height;j++) {
            data[i][j] = (float*) calloc(width,sizeof(float));
            for(int k=0;k<width;k++) {
                data[i][j][k] = random_normal(mean, std)*sqrt(2.0/n);
            }
        }
    }
    M->d = data;
    M->initialized = true;
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

int destroy3DMatrix (ThreeDMatrix* M) {
    for(int i = 0; i<M->depth;i++) {
        for(int j=0;j<M->height;j++) {
            free(M->d[i][j]);
            M->d[i][j] = NULL;
        }
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
            if (B->d[i][j] < 1e-6) {
                //printf("ERROR: A divide-by-0 exception is raised. A small bias of 1e-5 is added\n");
                OUT->d[i][j] = A->d[i][j] / (B->d[i][j]+1e-6);
            } else {
                OUT->d[i][j] = A->d[i][j] / B->d[i][j];
            }
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
    float n;
    if (a < 1e-6) {
        n = 1/(a+1e-6);
    } else {
        n = 1/a;
    }
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

int elementMul3DMatrix(ThreeDMatrix* X, float n, ThreeDMatrix* OUT) {
    init3DMatrix(OUT, X->depth, X->height, X->width);
    for (int i=0; i<OUT->depth; i++) {
        for(int j=0;j<OUT->height;j++) {
            for(int k=0;k<OUT->width;k++) {
                OUT->d[i][j][k] = n * X->d[i][j][k];
            }
        }
    }
    return 0;
}

int elementAdd3DMatrix(ThreeDMatrix* X, float n, ThreeDMatrix* OUT) {
    init3DMatrix(OUT, X->depth, X->height, X->width);
    for (int i=0; i<OUT->depth; i++) {
        for(int j=0;j<OUT->height;j++) {
            for(int k=0;k<OUT->width;k++) {
                OUT->d[i][j][k] = n + X->d[i][j][k];
            }
        }
    }
    return 0;
}

int elementSqrt3DMatrix(ThreeDMatrix* X, ThreeDMatrix* OUT) {
    init3DMatrix(OUT, X->depth, X->height, X->width);
    for (int i=0; i<OUT->depth; i++) {
        for(int j=0;j<OUT->height;j++) {
            for(int k=0;k<OUT->width;k++) {
                OUT->d[i][j][k] = sqrt(X->d[i][j][k]);
            }
        }
    }
    return 0;
}

int elementwiseAdd3DMatrix(ThreeDMatrix* A, ThreeDMatrix* B, ThreeDMatrix* OUT) {
    if (A->depth != B->depth || A->height != B->height || A->width != B->width) {
        printf("ERROR: Size mismatch while elementwise adding 3D matrixes\n");
        exit(1);
    }
    init3DMatrix(OUT, A->depth, A->height, A->width);
    for (int i=0; i<OUT->depth; i++) {
        for(int j=0;j<OUT->height;j++) {
            for(int k=0;k<OUT->width;k++) {
                OUT->d[i][j][k] = A->d[i][j][k] + B->d[i][j][k];
            }
        }
    }
    return 0;
}

int elementwiseSub3DMatrix(ThreeDMatrix* A, ThreeDMatrix* B, ThreeDMatrix* OUT) {
    if (A->depth != B->depth || A->height != B->height || A->width != B->width) {
        printf("ERROR: Size mismatch while elementwise adding 3D matrixes\n");
        exit(1);
    }
    init3DMatrix(OUT, A->depth, A->height, A->width);
    for (int i=0; i<OUT->depth; i++) {
        for(int j=0;j<OUT->height;j++) {
            for(int k=0;k<OUT->width;k++) {
                OUT->d[i][j][k] = A->d[i][j][k] - B->d[i][j][k];
            }
        }
    }
    return 0;
}

int elementwiseMul3DMatrix(ThreeDMatrix* A, ThreeDMatrix* B, ThreeDMatrix* OUT) {
    if (A->depth != B->depth || A->height != B->height || A->width != B->width) {
        printf("ERROR: Size mismatch while elementwise adding 3D matrixes\n");
        exit(1);
    }
    init3DMatrix(OUT, A->depth, A->height, A->width);
    for (int i=0; i<OUT->depth; i++) {
        for(int j=0;j<OUT->height;j++) {
            for(int k=0;k<OUT->width;k++) {
                OUT->d[i][j][k] = A->d[i][j][k] * B->d[i][j][k];
            }
        }
    }
    return 0;
}

int elementwiseDiv3DMatrix(ThreeDMatrix* A, ThreeDMatrix* B, ThreeDMatrix* OUT) {
    if (A->depth != B->depth || A->height != B->height || A->width != B->width) {
        printf("ERROR: Size mismatch while elementwise adding 3D matrixes\n");
        exit(1);
    }
    init3DMatrix(OUT, A->depth, A->height, A->width);
    for (int i=0; i<OUT->depth; i++) {
        for(int j=0;j<OUT->height;j++) {
            for(int k=0;k<OUT->width;k++) {
                OUT->d[i][j][k] = A->d[i][j][k] / B->d[i][j][k];
            }
        }
    }
    return 0;
}

ThreeDMatrix* chop3DMatrix(ThreeDMatrix* X, int start_y, int start_x, int end_y, int end_x) {
    int height = end_y - start_y + 1;
    int width = end_x - start_x + 1;
    ThreeDMatrix* out = matrixMalloc(sizeof(ThreeDMatrix));
    init3DMatrix(out, X->depth, height, width);
    for(int i=0;i<X->depth;i++) {
        for(int j=start_y;j<=end_y;j++) {
            for(int k=start_x;k<=end_x;k++) {
                out->d[i][j-start_y][k-start_x] = X->d[i][j][k];
            }
        }
    }
    return out;
}

int assign3DMatrix(ThreeDMatrix* in, int start_y, int start_x, int end_y, int end_x, ThreeDMatrix* X) {
    if (X->height < end_y || X->width < end_x) {
        printf("ERROR: Out of index in assign3DMatrix\n");
        exit(1);
    }
    for(int i=0;i<X->depth;i++) {
        for(int j=start_y;j<=end_y;j++) {
            for(int k=start_x;k<=end_x;k++) {
                X->d[i][j][k] = in->d[i][j-start_y][k-start_x];
            }
        }
    }
    return 0;
}

float decayLearningRate(bool enable_step_decay, bool enable_exponential_decay, bool enable_invert_t_decay, int decay_unit, float decay_k, float decay_a0, int epoch, float base_learning_rate, float learning_rate) {
    if (enable_step_decay) {
        if (epoch % decay_unit == 0) {
            return base_learning_rate*pow(decay_k,(epoch/decay_unit));
        }
    } else if (enable_exponential_decay) {
        if (epoch % decay_unit == 0) {
            return base_learning_rate*decay_a0*exp(-decay_k*(epoch/decay_unit));
        }
    } else if (enable_invert_t_decay) {
        if (epoch % decay_unit == 0) {
            return base_learning_rate*decay_a0/(1+decay_k*(epoch/decay_unit));
        }
    } 
    return learning_rate;
}

int normalize3DMatrixPerDepth(ThreeDMatrix* X, ThreeDMatrix* OUT) {
    init3DMatrix(OUT, X->depth, X->height, X->width);
    for(int i=0;i<X->depth;i++) {
        float mean = 0;
        for(int j=0;j<X->height;j++) {
            for(int k=0;k<X->width;k++) {
                mean += X->d[i][j][k];
            }
        }
        mean = mean / (X->height * X->width);
        float var = 0;
        for(int j=0;j<X->height;j++) {
            for(int k=0;k<X->width;k++) {
                var += (X->d[i][j][k] - mean)*(X->d[i][j][k] - mean);
            }
        }
        var = var / (X->height * X->width);
        float stddev = sqrt(var);
        if (stddev < 1e-6) {
            for(int j=0;j<X->height;j++) {
                for(int k=0;k<X->width;k++) {
                    OUT->d[i][j][k] = (X->d[i][j][k] - mean);
                }
            }
        } else {
            for(int j=0;j<X->height;j++) {
                for(int k=0;k<X->width;k++) {
                    OUT->d[i][j][k] = (X->d[i][j][k] - mean) / stddev;
                }
            }
        }
    }
    return 0;
}

int debugCheckingForNaNs2DMatrix(TwoDMatrix* X, char* name, int index) {
    for(int i=0;i<X->height;i++) {
        for(int j=0;j<X->width;j++) {
            if (isnan(X->d[i][j])) {
                printf("DEBUG: %s:%d %dx%d is nan\n", name, index, i, j);
                exit(1);
            }
        }
    }
    return 0;
}

int debugCheckingForNaNs3DMatrix(ThreeDMatrix* X, char* name, int index) {
    for(int i=0;i<X->depth;i++) {
        for(int j=0;j<X->height;j++) {
            for(int k=0;k<X->width;k++) {
                if (isnan(X->d[i][j][k])) {
                    printf("DEBUG: %s:%d %dx%dx%d is nan\n", name, index, i, j, k);
                    exit(1);
                }
            }
        }
    }
    return 0;
}


