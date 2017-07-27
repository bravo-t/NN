#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <stdbool.h>
#include "util/matrix_type.h"
#include "util/misc_utils.h"
#include "util/matrix_operations.h"
#include "util/layers.h"
#include "util/fully_connected_net.h"

void printMatrix(TwoDMatrix *M) {
    printf("Height of matrix: %d, width: %d\n",M->height,M->width);
    for(int i=0;i<M->height;i++) {
        for(int j=0;j<M->width;j++) {
            printf("%f\t",M->d[i][j]);
        }
        printf("\n");
    }
}

int loadTestData(char* filename, TwoDMatrix* training_data, TwoDMatrix* correct_labels) {
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
    init2DMatrix(training_data,height,width);
    for(int i=0;i<height;i++) {
        for(int j=0;j<width;j++) {
            fscanf(fp,"%f",&value);
            training_data->d[i][j] = value;
        }
    }
    fscanf(fp,"%s",buff);
    fscanf(fp,"%d",&height);
    fscanf(fp,"%d",&width);
    init2DMatrix(correct_labels,height,width);
    for(int i=0;i<height;i++) {
        for(int j=0;j<width;j++) {
            fscanf(fp,"%f",&value);
            correct_labels->d[i][j] = value;
        }
    }
    return 0;
}

int main() {
    TwoDMatrix* training_data = matrixMalloc(sizeof(TwoDMatrix));
    training_data = load2DMatrixFromFile("test_data/X.txt");
    TwoDMatrix* correct_labels = matrixMalloc(sizeof(TwoDMatrix));
    correct_labels = load2DMatrixFromFile("test_data/y.txt");
    printMatrix(training_data);
    printMatrix(correct_labels);
    destroy2DMatrix(training_data);
    destroy2DMatrix(correct_labels);
    return 0;
}
