#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <malloc.h>
#include "network_type.h"
#include "matrix_operations.h"
#include "convnet_operations.h"
#include "convnet_layers.h"
#include "convnet.h"

int trainConvnet(ConvnetParameters* network_params) {
    ThreeDMatrix** training_data = network_params->X;
    int number_of_samples = network_params->number_of_samples;
    /* INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC */
    int N = network_params->N;
    int M = network_params->M;
    int* filter_stride_x = network_params->filter_stride_x;
    int* filter_stride_y = network_params->filter_stride_y;
    int* filter_width = network_params->filter_width;
    int* filter_height = network_params->filter_height;
    int* filter_number = network_params->filter_number;
    bool* enable_maxpooling = network_params->enable_maxpooling;
    int* pooling_stride_x = network_params->pooling_stride_x;
    int* pooling_stride_y = network_params->pooling_stride_y;
    int* pooling_width = network_params->pooling_width;
    int* pooling_height = network_params->pooling_height;
    bool enable_padding = network_params->enable_padding;
    int padding_width = network_params->padding_width;
    int padding_height = network_params->padding_height;

    if (enable_padding) {
        for(int i=0;i<number_of_samples;i++) {
            ThreeDMatrix* tmp = matrixMalloc(sizeof(ThreeDMatrix));
            zeroPadding(training_data[i],padding_height,padding_width,tmp);
            destroy3DMatrix(training_data[i]);
            training_data[i] = tmp;
        }
    }

    printf("CONVNET INFO: Initializing learnable weights and intermediate layers\n");
    /*
    C will hold intermediate values of CONV -> RELU layer
    P will hold intermediate values of POOL
    F will be a 2D array that contains filters
    b will be a 2D array that holds biases
    */
    ThreeDMatrix**** C = malloc(sizeof(ThreeDMatrix***)*M);
    ThreeDMatrix**** dC = malloc(sizeof(ThreeDMatrix***)*M);
    ThreeDMatrix*** P = malloc(sizeof(ThreeDMatrix**)*M);
    ThreeDMatrix**** F = malloc(sizeof(ThreeDMatrix***)*M);
    ThreeDMatrix**** dF = malloc(sizeof(ThreeDMatrix***)*M);
    ThreeDMatrix**** b = malloc(sizeof(ThreeDMatrix***)*M);
    ThreeDMatrix**** db = malloc(sizeof(ThreeDMatrix***)*M);
    int layer_data_depth = training_data[0]->depth;
    int layer_data_height = training_data[0]->height;
    int layer_data_width = training_data[0]->width;
    printf("CONVNET INFO: INPUT: [%dx%dx%d]\t\tweights: 0\n",layer_data_width,layer_data_height,layer_data_depth);
    for(int i=0;i<M;i++) {
        C[i] = (ThreeDMatrix***) malloc(sizeof(ThreeDMatrix**)*N);
        F[i] = (ThreeDMatrix***) malloc(sizeof(ThreeDMatrix**)*N);
        b[i] = (ThreeDMatrix***) malloc(sizeof(ThreeDMatrix**)*N);
        dC[i] = (ThreeDMatrix***) malloc(sizeof(ThreeDMatrix**)*N);
        dF[i] = (ThreeDMatrix***) malloc(sizeof(ThreeDMatrix**)*N);
        db[i] = (ThreeDMatrix***) malloc(sizeof(ThreeDMatrix**)*N);
        for(int j=0;j<N;j++) {
            F[i][j] = (ThreeDMatrix**) malloc(sizeof(ThreeDMatrix*)*filter_number[i*M+j]);
            b[i][j] = (ThreeDMatrix**) malloc(sizeof(ThreeDMatrix*)*filter_number[i*M+j]);
            C[i][j] = (ThreeDMatrix**) malloc(sizeof(ThreeDMatrix*)*number_of_samples);
            dF[i][j] = (ThreeDMatrix**) malloc(sizeof(ThreeDMatrix*)*filter_number[i*M+j]);
            db[i][j] = (ThreeDMatrix**) malloc(sizeof(ThreeDMatrix*)*filter_number[i*M+j]);
            dC[i][j] = (ThreeDMatrix**) malloc(sizeof(ThreeDMatrix*)*number_of_samples);
            for(int k=0;k<filter_number[i*M+j];k++) {
                init3DMatrixNormRand(F[i][j][k],layer_data_depth,filter_height[i*M+j],filter_width[i*M+j],0.0,1.0);
                init3DMatrix(b[i][j][k],1,1,1);
                init3DMatrix(dF[i][j][k],layer_data_depth,filter_height[i*M+j],filter_width[i*M+j]);
                init3DMatrix(db[i][j][k],1,1,1);
            }
            int filter_depth = layer_data_depth;
            layer_data_depth = filter_number[i*M+j];
            layer_data_height = calcOutputSize(layer_data_height,0,filter_height[i*M+j],filter_stride_y[i*M+j]);
            layer_data_width = calcOutputSize(layer_data_width,0,filter_width[i*M+j],filter_stride_x[i*M+j]);
            for(int l=0;l<number_of_samples;l++) {
                init3DMatrix(C[i][j][l], layer_data_depth, layer_data_height, layer_data_width);
                init3DMatrix(dC[i][j][l], layer_data_depth, layer_data_height, layer_data_width);
            }
            printf("CONVNET INFO: CONV[%dx%d-%d]: [%dx%dx%d]\t\tweights: (%d*%d*%d)*%d=%d\n",
                filter_width[i*M+j],filter_height[i*M+j],filter_depth, layer_data_width,layer_data_height,layer_data_depth,
                filter_width[i*M+j],filter_height[i*M+j],filter_depth,layer_data_depth,filter_width[i*M+j]*filter_height[i*M+j]*filter_depth*layer_data_depth);
        }
        if (enable_maxpooling[i]) {
            P[i] = (ThreeDMatrix**) malloc(sizeof(ThreeDMatrix*)*number_of_samples);
            layer_data_height = calcOutputSize(layer_data_height,0,pooling_height[i],pooling_stride_y[i]);
            layer_data_width = calcOutputSize(layer_data_width,0,pooling_width[i],pooling_stride_x[i]);
            for(int m=0;m<number_of_samples;m++) {
                init3DMatrix(P[i][m],layer_data_depth,layer_data_height,layer_data_width);
            }
            printf("CONVNET INFO: POOL[%dx%d]: [%dx%dx%d]\t\tweights: 0\n",pooling_width[i],pooling_height[i],layer_data_width,layer_data_height,layer_data_depth);
        } else {
            P[i] = NULL;
        }
    }
}
