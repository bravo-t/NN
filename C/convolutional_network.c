#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <malloc.h>
#include <math.h>
#include <string.h>
#include "src/network_type.h"
#include "src/matrix_operations.h"
#include "src/layers.h"
#include "src/misc_utils.h"
#include "src/fully_connected_net.h"
#include "src/convnet_operations.h"
#include "src/convnet_layers.h"
#include "src/convnet.h"

int main(int argc, char** argv) {
    if (argc == 1) {
        printf("Please specify config file\n");
        exit(0);
    }
    ConvnetParameters* network_params = readConvnetConfigFile(argv[1]);
    if (strcmp(network_params->mode,"train")) {
    	TwoDMatrix* scores = NULL;
        testConvnet(network_params,scores);
    } else {
        trainConvnet(network_params);
    }
}
