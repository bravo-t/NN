#include <malloc.h>
#include <stdbool.h>
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
    init2DMatrix(training_data,height,width);
    for(int i=0;i<height;i++) {
        for(int j=0;j<width;j++) {
            fscanf(fp,"%f",&value);
            training_data->d[i][j] = value;
        }
    }
    fclose(fp);
    return M;
}


