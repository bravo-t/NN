#include <malloc.h>
#include <stdbool.h>
#include <stdlib.h>
#include "matrix_type.h"
#include "misc_utils.h"

TwoDMatrix* matrixMalloc(int size) {
    TwoDMatrix* M = malloc(size);
    M->initialized = false;
    return M;
}

int dumpLearnableParams(TwoDMatrix** Ws, TwoDMatrix** bs) {
    return 0;
}

int loadLearnableParams(TwoDMatrix** Ws, TwoDMatrix** bs) {
    return 0;
}

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
    float sum_a = sumAll(a);
    float sum_b = sumAll(b);
    return sum_a - sum_b;
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

void checkMatrixDiff(TwoDMatrix* a, TwoDMatrix* b, float thres) {
    float diff = matrixError(a, b);
    if (diff >= thres) {
        printf("ERROR: Too much differences between ref and impl\n");
        printf("ref = \n");
        printMatrix(a);
        printf("impl = \n");
        printMatrix(b);
    }
}
