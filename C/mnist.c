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

const int i = 1;
#define is_bigendian() ( (*(char*)&i) == 0)

ThreeDMatrix** readMNISTImageSet(char* filename);
TwoDMatrix* readMNISTLabels(char* filename);
unsigned int convertIntEndian(unsigned int a);


int main() {
    ThreeDMatrix** training_data = readMNISTImageSet("../mnist/train-images-idx3-ubyte");
    TwoDMatrix* correct_labels = readMNISTLabels("../mnist/train-labels-idx1-ubyte");
    ConvnetParameters* convnet_params = malloc(sizeof(ConvnetParameters));
    memset(convnet_params, 0, sizeof(ConvnetParameters));
    convnet_params->fcnet_param = (FCParameters*) malloc(sizeof(FCParameters));
    convnet_params->X = training_data;
    convnet_params->number_of_samples = 1000;
    convnet_params->minibatch_size = 50;
    //convnet_params->number_of_samples = number_of_batch_data*10000;
    convnet_params->M = 3;
    convnet_params->N = 1;
    int stride[3] = {1, 2,1};
    int filter_size[3] = {14,7,3};
    int filter_number[3] = {8,16,16};
    bool enable_maxpooling[3] = {false,false,false};
    int maxpooling_stride[3] = {2,3};
    int maxpooling_size[3] = {2,3};
    int padding[3] = {0,0,0};
    convnet_params->filter_stride_x = stride;
    convnet_params->filter_stride_y = stride;
    convnet_params->filter_width = filter_size;
    convnet_params->filter_height = filter_size;
    convnet_params->filter_number = filter_number;
    convnet_params->enable_maxpooling = enable_maxpooling;
    convnet_params->pooling_height = maxpooling_size;
    convnet_params->pooling_width = maxpooling_size;
    convnet_params->pooling_stride_x = maxpooling_stride;
    convnet_params->pooling_stride_y = maxpooling_stride;
    convnet_params->padding_height = padding;
    convnet_params->padding_width = padding;
    convnet_params->epochs = 2000;
    convnet_params->alpha = 1e-3;
    convnet_params->learning_rate = 0.001;
    convnet_params->verbose = false;
    convnet_params->use_rmsprop = true;
    convnet_params->rmsprop_decay_rate = 0.9;
    convnet_params->rmsprop_eps = 1e-5;
    convnet_params->fcnet_param->use_rmsprop = true;
    convnet_params->fcnet_param->decay_rate = 0.9;
    convnet_params->fcnet_param->eps = 1e-5;
    convnet_params->normalize_data_per_channel = true;
    convnet_params->write_filters_as_images = true;
    convnet_params->filter_image_dir = "img";
    convnet_params->enable_learning_rate_step_decay = true;
    convnet_params->learning_rate_decay_unit = 250;
    convnet_params->learning_rate_decay_a0 = 1.0;
    convnet_params->learning_rate_decay_k = 0.9;
    convnet_params->fcnet_param->correct_labels = correct_labels;
    int hidden_layer_sizes[2] = {50, 10};
    convnet_params->fcnet_param->hidden_layer_sizes = hidden_layer_sizes;
    convnet_params->fcnet_param->labels = 10;
    convnet_params->fcnet_param->network_depth = 2;
    convnet_params->fcnet_param->reg_strength = 50;
    convnet_params->fcnet_param->learning_rate = convnet_params->learning_rate;
    trainConvnet(convnet_params);
    return 0;
}

ThreeDMatrix** readMNISTImageSet(char* filename) {
    FILE* fp = fopen(filename,"rb");
    if (fp == NULL) {
        printf("ERROR: Cannot open file %s\n", filename);
        exit(1);
    }
    unsigned int header;
    fread(&header,4,1,fp);
    header = convertIntEndian(header);
    if (header != 2051) {
        printf("Magic number is wrong. Should be 0x0803, got 0x%x\n",header);
        printf("ERROR: %s does not have the correct contents\n", filename);
        exit(1);
    }
    unsigned int number_of_samples;
    fread(&number_of_samples,4,1,fp);
    number_of_samples = convertIntEndian(number_of_samples);
    ThreeDMatrix** X = malloc(sizeof(ThreeDMatrix*)*number_of_samples);
    unsigned int width,height;
    fread(&height,4,1,fp);
    fread(&width,4,1,fp);
    width = convertIntEndian(width);
    height = convertIntEndian(height);
    for(int i=0;i<number_of_samples;i++) {
        X[i] = matrixMalloc(sizeof(ThreeDMatrix));
        init3DMatrix(X[i],1,height,width);
    }
    for(int i=0;i<number_of_samples;i++) {
        unsigned char buffer;
        for(int j=0;j<height;j++) {
            for(int k=0;k<width;k++) {
                fread(&buffer,1,1,fp);
                X[i]->d[0][j][k] = buffer;
            }
        }
    }
    fclose(fp);
    return X;
}

TwoDMatrix* readMNISTLabels(char* filename) {
    FILE* fp = fopen(filename,"rb");
    if (fp == NULL) {
        printf("ERROR: Cannot open file %s\n", filename);
        exit(1);
    }
    unsigned int header;
    fread(&header,4,1,fp);
    header = convertIntEndian(header);
    if (header != 2049) {
        printf("ERROR: %s does not have the correct contents\n", filename);
        exit(1);
    }
    unsigned int number_of_samples;
    fread(&number_of_samples,4,1,fp);
    number_of_samples = convertIntEndian(number_of_samples);
    TwoDMatrix* correct_labels = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(correct_labels,number_of_samples,1);
    for(int i=0;i<number_of_samples;i++) {
        unsigned char buffer;
        fread(&buffer,1,1,fp);
            correct_labels->d[i][0] = buffer;
    }
    fclose(fp);
    return correct_labels;
}

unsigned int convertIntEndian(unsigned int a) {
    unsigned int b0,b1,b2,b3;
    b0 = (a & 0x000000ff) << 24u;
    b1 = (a & 0x0000ff00) << 8u;
    b2 = (a & 0x00ff0000) >> 8u;
    b3 = (a & 0xff000000) >> 24u;
    return (b0 | b1 | b2 | b3);
}
