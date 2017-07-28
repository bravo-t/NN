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
    TwoDMatrix* W = matrixMalloc(sizeof(TwoDMatrix));
    W = load2DMatrixFromFile("test_data/W.txt");
    TwoDMatrix* b = matrixMalloc(sizeof(TwoDMatrix));
    b = load2DMatrixFromFile("test_data/b.txt");
    TwoDMatrix* W2 = matrixMalloc(sizeof(TwoDMatrix));
    W2 = load2DMatrixFromFile("test_data/W2.txt");
    TwoDMatrix* b2 = matrixMalloc(sizeof(TwoDMatrix));
    b2 = load2DMatrixFromFile("test_data/b2.txt");
    TwoDMatrix* ref_H_before_relu = matrixMalloc(sizeof(TwoDMatrix));
    ref_H_before_relu = load2DMatrixFromFile("test_data/H_before_relu.txt");
    TwoDMatrix* ref_H_after_relu = matrixMalloc(sizeof(TwoDMatrix));
    ref_H_after_relu = load2DMatrixFromFile("test_data/H_after_relu.txt");
    TwoDMatrix* ref_softmax_score = matrixMalloc(sizeof(TwoDMatrix));
    ref_softmax_score = load2DMatrixFromFile("test_data/softmax_score.txt");
    TwoDMatrix* ref_dscore = matrixMalloc(sizeof(TwoDMatrix));
    ref_dscore = load2DMatrixFromFile("test_data/dscore.txt");
    TwoDMatrix* ref_dW2 = matrixMalloc(sizeof(TwoDMatrix));
    ref_dW2 = load2DMatrixFromFile("test_data/dW2.txt");
    TwoDMatrix* ref_db2 = matrixMalloc(sizeof(TwoDMatrix));
    ref_db2 = load2DMatrixFromFile("test_data/db2.txt");
    TwoDMatrix* ref_dH_before_relu = matrixMalloc(sizeof(TwoDMatrix));
    ref_dH_before_relu = load2DMatrixFromFile("test_data/dH_before_relu.txt");
    TwoDMatrix* ref_dH_after_relu = matrixMalloc(sizeof(TwoDMatrix));
    ref_dH_after_relu = load2DMatrixFromFile("test_data/dH_after_relu.txt");
    TwoDMatrix* ref_dW = matrixMalloc(sizeof(TwoDMatrix));
    ref_dW = load2DMatrixFromFile("test_data/dW.txt");
    TwoDMatrix* ref_db = matrixMalloc(sizeof(TwoDMatrix));
    ref_db = load2DMatrixFromFile("test_data/db.txt");
    TwoDMatrix* ref_dW2_after_reg_back = matrixMalloc(sizeof(TwoDMatrix));
    ref_dW2_after_reg_back = load2DMatrixFromFile("test_data/dW2_after_reg_back.txt");
    TwoDMatrix* ref_dW_after_reg_back = matrixMalloc(sizeof(TwoDMatrix));
    ref_dW_after_reg_back = load2DMatrixFromFile("test_data/dW_after_reg_back.txt");
    TwoDMatrix* ref_W_after_update = matrixMalloc(sizeof(TwoDMatrix));
    ref_W_after_update = load2DMatrixFromFile("test_data/W_after_update.txt");
    TwoDMatrix* ref_b_after_update = matrixMalloc(sizeof(TwoDMatrix));
    ref_b_after_update = load2DMatrixFromFile("test_data/b_after_update.txt");
    TwoDMatrix* ref_W2_after_update = matrixMalloc(sizeof(TwoDMatrix));
    ref_W2_after_update = load2DMatrixFromFile("test_data/W2_after_update.txt");
    TwoDMatrix* ref_b2_after_update = matrixMalloc(sizeof(TwoDMatrix));
    ref_b2_after_update = load2DMatrixFromFile("test_data/b2_after_update.txt");
    TwoDMatrix* H = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* score = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* H_before_relu = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* H_after_relu = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* dscore = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* dW2 = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* dW = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* db = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* db2 = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* dH_before_relu = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* dH_after_relu = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* dH = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* dW2_after_reg_back = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* dW_after_reg_back = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* W_after_update = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* b_after_update = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* W2_after_update = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* b2_after_update = matrixMalloc(sizeof(TwoDMatrix));
    destroy2DMatrix(training_data);
    destroy2DMatrix(correct_labels);
    return 0;
}
