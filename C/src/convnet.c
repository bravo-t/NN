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
    bool* enable_maxpooling = network_params->enable_maxpooling;
    int* pooling_stride_x = network_params->pooling_stride_x;
    int* pooling_stride_y = network_params->pooling_stride_y;
    int* pooling_width = network_params->pooling_width;
    int* pooling_height = network_params->pooling_height;

    printf("CONVNET INFO: Initializing learnable weights ang intermediate layers\n");
    /*
    C will hold intermediate values of CONV -> RELU layer
    P will hold intermediate values of POOL
    F will be a 2D array that contains filters
    */
    ThreeDMatrix*** C = malloc(sizeof(ThreeDMatrix**)*N*M);
    for(int i=0;i<M*N;i++) {
        C[i] = (ThreeDMatrix**) malloc(sizeof(ThreeDMatrix*)*number_of_samples);
    }
    ThreeDMatrix*** P = malloc(sizeof(ThreeDMatrix**)*M);
    for(int i=0;i<M;i++) {
        P[i] = (ThreeDMatrix**) malloc(sizeof(ThreeDMatrix*)*number_of_samples);
    }
    ThreeDMatrix*** F = malloc(sizeof(ThreeDMatrix**)*N*M);
    for(int i=0;i<M;i++) {
        for(int j=0;j<N;j++) {
            
        }
    }
}