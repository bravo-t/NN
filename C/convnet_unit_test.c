#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <malloc.h>
#include <string.h>
#include "src/network_type.h"
#include "src/matrix_operations.h"
#include "src/layers.h"
#include "src/misc_utils.h"
#include "src/fully_connected_net.h"
#include "src/convnet_operations.h"
#include "src/convnet_layers.h"
#include "src/convnet.h"

int main(int argc, char const *argv[])
{
    float thres = 1e-6;
    ThreeDMatrix* x = load3DMatrixFromFile("test_data/convnet_x.txt");
    ThreeDMatrix** f = malloc(sizeof(ThreeDMatrix*));
    ThreeDMatrix** b = malloc(sizeof(ThreeDMatrix*));
    ThreeDMatrix* ref_out_after_conv_forward = load3DMatrixFromFile("test_data/convnet_out_after_conv_forward.txt");
    ThreeDMatrix* out_after_conv_forward = matrixMalloc(sizeof(ThreeDMatrix));
    f[0] = (ThreeDMatrix*) load3DMatrixFromFile("test_data/convnet_f.txt");
    b[0] = (ThreeDMatrix*) load3DMatrixFromFile("test_data/convnet_b.txt");
    printf("x\n");
    print3DMatrix(x);
    convLayerForward(x, 
        f, 
        1, 
        b, 
        2, 
        2, 
        2, 
        2, 
        0, 
        0, 
        0.0f, 
        out_after_conv_forward);
    printf("Comparing out_after_conv_forward\n");
    check3DMatrixDiff(ref_out_after_conv_forward, out_after_conv_forward, thres);

    ThreeDMatrix* padded = matrixMalloc(sizeof(ThreeDMatrix));
    ThreeDMatrix* unpadded = matrixMalloc(sizeof(ThreeDMatrix));
    zeroPadding(ref_out_after_conv_forward, 2, 2, padded);
    unpad(padded, 2, 2, unpadded);
    printf("ref_out_after_conv_forward\n");
    print3DMatrix(ref_out_after_conv_forward);
    printf("padded\n");
    print3DMatrix(padded);
    printf("unpadded\n");
    print3DMatrix(unpadded);

    ThreeDMatrix* dout = load3DMatrixFromFile("test_data/convnet_dout_before_conv_backward.txt");
    ThreeDMatrix* dx_after_conv_backward = matrixMalloc(sizeof(ThreeDMatrix));
    ThreeDMatrix* ref_dx_after_conv_backward = load3DMatrixFromFile("test_data/convnet_dx_after_conv_backward.txt");
    ThreeDMatrix* ref_df_after_conv_backward = load3DMatrixFromFile("test_data/convnet_dw_after_conv_backward.txt");
    ThreeDMatrix* ref_db_after_conv_backward = load3DMatrixFromFile("test_data/convnet_db_after_conv_backward.txt");
    ThreeDMatrix** df_after_conv_backward = malloc(sizeof(ThreeDMatrix*));
    ThreeDMatrix** db_after_conv_backward = malloc(sizeof(ThreeDMatrix*));
    df_after_conv_backward[0] = (ThreeDMatrix*) matrixMalloc(sizeof(ThreeDMatrix));
    db_after_conv_backward[0] = (ThreeDMatrix*) matrixMalloc(sizeof(ThreeDMatrix));
    init3DMatrix(db_after_conv_backward[0],1,1,1);
    convLayerBackward(x, 
        ref_out_after_conv_forward,
        f, 
        dout, 
        0, 
        0, 
        2, 
        2, 
        0.0f,
        dx_after_conv_backward, 
        df_after_conv_backward, 
        db_after_conv_backward);
    printf("Comparing dx_after_conv_backward\n");
    check3DMatrixDiff(ref_dx_after_conv_backward, dx_after_conv_backward, thres);
    printf("Comparing df_after_conv_backward\n");
    check3DMatrixDiff(ref_df_after_conv_backward, df_after_conv_backward[0], thres);
    printf("Comparing db_after_conv_backward\n");
    check3DMatrixDiff(ref_db_after_conv_backward, db_after_conv_backward[0], thres);

    ThreeDMatrix* pool_out_before_pool_foreward = matrixMalloc(sizeof(ThreeDMatrix));
    ThreeDMatrix* ref_pool_out_before_pool_foreward = load3DMatrixFromFile("test_data/convnet_pool_out_before_pool_foreward.txt");
    maxPoolingForward(ref_out_after_conv_forward, 
        2, 
        2, 
        2, 
        2, 
        pool_out_before_pool_foreward);
    printf("Comparing pool_out_before_pool_foreward\n");
    check3DMatrixDiff(ref_pool_out_before_pool_foreward, pool_out_before_pool_foreward, thres);

    ThreeDMatrix* pool_dout = load3DMatrixFromFile("test_data/convnet_pool_dout_before_pool_backward.txt");
    ThreeDMatrix* pool_dx_after_pool_backward = matrixMalloc(sizeof(ThreeDMatrix));
    ThreeDMatrix* ref_pool_dx_after_pool_backward = load3DMatrixFromFile("test_data/convnet_pool_dx_after_pool_backward.txt");
    maxPoolingBackword(pool_dout, 
        ref_out_after_conv_forward, 
        2, 
        2, 
        2, 
        2, 
        pool_dx_after_pool_backward);
    printf("Comparing pool_dx_after_pool_backward\n");
    check3DMatrixDiff(ref_pool_dx_after_pool_backward, pool_dx_after_pool_backward, thres);
    return 0;

}