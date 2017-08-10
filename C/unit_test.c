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
    float thres = 1e-6;
    float reg = 1e-3;
    float learning_rate = 1;
    TwoDMatrix** Ws = malloc(sizeof(TwoDMatrix*)*2);
    TwoDMatrix* X = matrixMalloc(sizeof(TwoDMatrix));
    X = load2DMatrixFromFile("test_data/X.txt");
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
    
    // hidden_before_relu = np.dot(X,W) + b
    TwoDMatrix* H = matrixMalloc(sizeof(TwoDMatrix));
    affineLayerForward(X,W,b,H);
    TwoDMatrix* H_before_relu = matrixMalloc(sizeof(TwoDMatrix));
    printf("Comparing H_before_relu\n");
    affineLayerForward(X,W,b,H_before_relu);
    checkMatrixDiff(ref_H_before_relu,H_before_relu,thres);
    
    // hidden = np.maximum(0,hidden_before_relu)
    TwoDMatrix* H_after_relu = matrixMalloc(sizeof(TwoDMatrix));
    printf("Comparing H_after_relu\n");
    leakyReLUForward(ref_H_before_relu,0,H_after_relu);
    checkMatrixDiff(ref_H_after_relu,H_after_relu,thres);
    
    printf("Comparing self H after relu\n");
    leakyReLUForward(H,0,H);
    checkMatrixDiff(ref_H_after_relu,H,thres);

    // score = np.dot(hidden,W2) + b2
    TwoDMatrix* score = matrixMalloc(sizeof(TwoDMatrix));
    printf("Comparing softmax score\n");
    affineLayerForward(ref_H_after_relu,W2,b2,score);
    checkMatrixDiff(ref_softmax_score,score,thres);

    /*
     * score_exp = np.exp(score)
    probs = score_exp / np.sum(score_exp,axis=1,keepdims=True)
    correct_loss = probs[range(data_size),y]
    logged_loss = -np.log(correct_loss)
    data_loss = np.sum(logged_loss) / data_size
    dscore = probs
    dscore[range(data_size),y] -= 1
    dscore /= data_size
    */
    Ws[0] = W;
    Ws[1] = W2;
    TwoDMatrix* dscore = matrixMalloc(sizeof(TwoDMatrix));
    printf("Comparing softmax loss and dscore\n");
    float data_loss = softmaxLoss(ref_softmax_score, correct_labels, dscore);
    printf("Expecting data loss: 1.098627, got %f\n", data_loss);
    checkMatrixDiff(ref_dscore,dscore,thres);
    float reg_loss = L2RegLoss(Ws, 2, reg);
    printf("Expecting reg loss: 0.000001, got %f\n", reg_loss);
    // dW2 = np.dot(hidden.T,dscore)
    // db2 = np.sum(dscore,axis=0,keepdims=True)
    // dhidden = np.dot(dscore,W2.T)
    TwoDMatrix* dW2 = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* dW = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* db = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* db2 = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* dH_before_relu = matrixMalloc(sizeof(TwoDMatrix));
    affineLayerBackword(ref_dscore,ref_H_after_relu,W2,b2,dH_before_relu,dW2,db2);
    printf("Comparing dH_before_relu\n");
    checkMatrixDiff(ref_dH_before_relu,dH_before_relu,thres);
    printf("Comparing dW2\n");
    checkMatrixDiff(ref_dW2,dW2,thres);

    // dhidden[hidden <= 0] = 0
    TwoDMatrix* dH_after_relu = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* dH = matrixMalloc(sizeof(TwoDMatrix));
    copyTwoDMatrix(ref_dH_before_relu,dH);
    printf("Comparing dH_after_relu\n");
    printf("ref_dH_before_relu = \n");
    printMatrix(ref_dH_before_relu);
    printf("ref_H_after_relu = \n");
    printMatrix(ref_H_after_relu);
    leakyReLUBackward(ref_dH_before_relu,ref_H_after_relu,0,dH_after_relu);
    checkMatrixDiff(ref_dH_after_relu,dH_after_relu,thres);

    printf("Comparing self dH\n");
    leakyReLUBackward(dH,ref_H_after_relu,0,dH);
    checkMatrixDiff(ref_dH_after_relu,dH,thres);

    TwoDMatrix* dW2_after_reg_back = matrixMalloc(sizeof(TwoDMatrix));
    printf("Comparing dW2_after_reg_back\n");
    L2RegLossBackward(ref_dW2,W2,reg,dW2_after_reg_back);
    checkMatrixDiff(ref_dW2_after_reg_back,dW2_after_reg_back,thres);
    copyTwoDMatrix(ref_dW2,dW2);
    printf("Comparing self dW2\n");
    L2RegLossBackward(dW2,W2,reg,dW2);
    checkMatrixDiff(ref_dW2_after_reg_back,dW2,thres);

    TwoDMatrix* dW_after_reg_back = matrixMalloc(sizeof(TwoDMatrix));
    printf("Comparing dW_after_reg_back\n");
    L2RegLossBackward(ref_dW,W,reg,dW_after_reg_back);
    checkMatrixDiff(ref_dW_after_reg_back,dW_after_reg_back,thres);
    copyTwoDMatrix(ref_dW,dW);
    printf("Comparing self dW\n");
    L2RegLossBackward(dW,W,reg,dW);
    checkMatrixDiff(ref_dW_after_reg_back,dW,thres);

    TwoDMatrix* W_after_update = matrixMalloc(sizeof(TwoDMatrix));
    printf("Comparing W_after_update\n");
    vanillaUpdate(W,dW,learning_rate,W_after_update);
    checkMatrixDiff(ref_W_after_update,W_after_update,thres);
    TwoDMatrix* W_self = matrixMalloc(sizeof(TwoDMatrix));
    copyTwoDMatrix(W,W_self);
    printf("Comparing self W_after_update\n");
    vanillaUpdate(W_self,dW,learning_rate,W_self);
    checkMatrixDiff(ref_W_after_update,W_self,thres);

    TwoDMatrix* b_after_update = matrixMalloc(sizeof(TwoDMatrix));
    printf("Comparing b_after_update\n");
    vanillaUpdate(b,ref_db,learning_rate,b_after_update);
    checkMatrixDiff(ref_b_after_update,b_after_update,thres);
    TwoDMatrix* b_self = matrixMalloc(sizeof(TwoDMatrix));
    copyTwoDMatrix(b,b_self);
    printf("Comparing self b_after_update\n");
    vanillaUpdate(b_self,ref_db,learning_rate,b_self);
    checkMatrixDiff(ref_b_after_update,b_self,thres);

    TwoDMatrix* W2_after_update = matrixMalloc(sizeof(TwoDMatrix));
    printf("Comparing W2_after_update\n");
    vanillaUpdate(W2,dW2,learning_rate,W2_after_update);
    checkMatrixDiff(ref_W2_after_update,W2_after_update,thres);
    TwoDMatrix* W2_self = matrixMalloc(sizeof(TwoDMatrix));
    copyTwoDMatrix(W2,W2_self);
    printf("Comparing self W2_after_update\n");
    vanillaUpdate(W2_self,dW2,learning_rate,W2_self);
    checkMatrixDiff(ref_W2_after_update,W2_self,thres);

    TwoDMatrix* b2_after_update = matrixMalloc(sizeof(TwoDMatrix));
    printf("Comparing b2_after_update\n");
    vanillaUpdate(b2,db2,learning_rate,b2_after_update);
    checkMatrixDiff(ref_b2_after_update,b2_after_update,thres);
    TwoDMatrix* b2_self = matrixMalloc(sizeof(TwoDMatrix));
    copyTwoDMatrix(b2,b2_self);
    printf("Comparing self b2_after_update\n");
    vanillaUpdate(b2_self,db2,learning_rate,b2_self);
    checkMatrixDiff(ref_b2_after_update,b2_self,thres);

    TwoDMatrix* H_before_batchnorm_forward = matrixMalloc(sizeof(TwoDMatrix));
    H_before_batchnorm_forward = load2DMatrixFromFile("test_data/H_before_batchnorm_forward.txt");
    TwoDMatrix* gamma_before_batchnorm_forward = matrixMalloc(sizeof(TwoDMatrix));
    gamma_before_batchnorm_forward = load2DMatrixFromFile("test_data/gamma_before_batchnorm_forward.txt");
    TwoDMatrix* beta_before_batchnorm_forward = matrixMalloc(sizeof(TwoDMatrix));
    beta_before_batchnorm_forward = load2DMatrixFromFile("test_data/beta_before_batchnorm_forward.txt");
    TwoDMatrix* mean_caches_before_batchnorm_forward = matrixMalloc(sizeof(TwoDMatrix));
    mean_caches_before_batchnorm_forward = load2DMatrixFromFile("test_data/beta_before_batchnorm_forward.txt");
    TwoDMatrix* var_caches_before_batchnorm_forward = matrixMalloc(sizeof(TwoDMatrix));
    var_caches_before_batchnorm_forward = load2DMatrixFromFile("test_data/beta_before_batchnorm_forward.txt");

    TwoDMatrix* H_after_batchnorm_forward = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* sample_mean_after_batchnorm_forward = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* sample_var_after_batchnorm_forward = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* H_normalized_after_batchnorm_forward = matrixMalloc(sizeof(TwoDMatrix));
    TwoDMatrix* ref_H_after_batchnorm_forward = matrixMalloc(sizeof(TwoDMatrix));
    ref_H_after_batchnorm_forward = load2DMatrixFromFile("test_data/H_after_batchnorm_forward.txt");
    TwoDMatrix* ref_H_normalized_after_batchnorm_forward = matrixMalloc(sizeof(TwoDMatrix));
    ref_H_normalized_after_batchnorm_forward = load2DMatrixFromFile("test_data/H_normalized_after_batchnorm_forward.txt");
    TwoDMatrix* ref_sample_mean_after_batchnorm_forward = matrixMalloc(sizeof(TwoDMatrix));
    ref_sample_mean_after_batchnorm_forward = load2DMatrixFromFile("test_data/mean_after_batchnorm_forward.txt");
    TwoDMatrix* ref_sample_var_after_batchnorm_forward = matrixMalloc(sizeof(TwoDMatrix));
    ref_sample_var_after_batchnorm_forward = load2DMatrixFromFile("test_data/var_after_batchnorm_forward.txt");
    TwoDMatrix* ref_mean_caches_after_batchnorm_forward = matrixMalloc(sizeof(TwoDMatrix));
    ref_mean_mean_caches_after_batchnorm_forward = load2DMatrixFromFile("test_data/mean_after_batchnorm_forward.txt");
    TwoDMatrix* ref_var_caches_after_batchnorm_forward = matrixMalloc(sizeof(TwoDMatrix));
    ref_var_caches_after_batchnorm_forward = load2DMatrixFromFile("test_data/var_after_batchnorm_forward.txt");
//    TwoDMatrix* ref__after_batchnorm_forward = matrixMalloc(sizeof(TwoDMatrix));
//    ref__after_batchnorm_forward = load2DMatrixFromFile("test_data/_after_batchnorm_forward.txt");
//    TwoDMatrix* ref__after_batchnorm_forward = matrixMalloc(sizeof(TwoDMatrix));
//    ref__after_batchnorm_forward = load2DMatrixFromFile("test_data/_after_batchnorm_forward.txt");


    batchnorm_training_forward(H_before_batchnorm_forward, 
        0.0, 
        1e-5, 
        gamma_before_batchnorm_forward, 
        beta_before_batchnorm_forward, 
        H_after_batchnorm_forward, 
        mean_caches_before_batchnorm_forward, 
        var_caches_before_batchnorm_forward, 
        sample_mean_after_batchnorm_forward, 
        sample_var_after_batchnorm_forward, 
        H_normalized_after_batchnorm_forward);

    printf("Comparing H_after_batchnorm_forward\n");
    checkMatrixDiff(ref_H_after_batchnorm_forward,H_after_batchnorm_forward);
    printf("Comparing mean_caches_after_batchnorm_forward\n");
    checkMatrixDiff(ref_mean_caches_after_batchnorm_forward,mean_caches_after_batchnorm_forward);
    printf("Comparing var_caches_after_batchnorm_forward\n");
    checkMatrixDiff(ref_var_caches_after_batchnorm_forward,var_caches_after_batchnorm_forward);
    printf("Comparing sample_mean_after_batchnorm_forward\n");
    checkMatrixDiff(ref_sample_mean_after_batchnorm_forward,sample_mean_after_batchnorm_forward);
    printf("Comparing sample_var_after_batchnorm_forward\n");
    checkMatrixDiff(ref_sample_var_after_batchnorm_forward,sample_var_after_batchnorm_forward);
    printf("Comparing H_normalized_after_batchnorm_forward\n");
    checkMatrixDiff(ref_H_normalized_after_batchnorm_forward,H_normalized_after_batchnorm_forward);

    TwoDMatrix* dH_before_batchnorm_backward = matrixMalloc(sizeof(TwoDMatrix));
    dH_before_batchnorm_backward = load2DMatrixFromFile("test_data/dH_before_batchnorm_backward.txt");
    TwoDMatrix* ref_dgamma_after_batchnorm_backward = matrixMalloc(sizeof(TwoDMatrix));
    ref_dgamma_after_batchnorm_backward = load2DMatrixFromFile("test_data/dgamma.txt");
    TwoDMatrix* ref_dbeta_after_batchnorm_backward = matrixMalloc(sizeof(TwoDMatrix));
    ref_dbeta_after_batchnorm_backward = load2DMatrixFromFile("test_data/dbeta.txt");
    batchnorm_backward(TwoDMatrix* dOUT, 
        TwoDMatrix* M, 
        TwoDMatrix* M_normalized, 
        TwoDMatrix* gamma, 
        TwoDMatrix* beta, 
        TwoDMatrix* mean, 
        TwoDMatrix* var, 
        float eps, 
        TwoDMatrix* dM, 
        TwoDMatrix* dgamma, 
        TwoDMatrix* dbeta);


    destroy2DMatrix(X);
    destroy2DMatrix(correct_labels);
    return 0;
}
