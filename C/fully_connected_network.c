#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <stdbool.h>
#include <string.h>
#include "src/network_type.h"
#include "src/matrix_operations.h"
#include "src/layers.h"
#include "src/fully_connected_net.h"
#include "src/misc_utils.h"

int main(int argc, char** argv) {
    if (argc == 1) {
        printf("Please specify config file\n");
        exit(0);
    }
    FCParameters* network_params = readNetworkConfigFile(argv[1]);
    if (strcmp(network_params->mode,"train")) {
        test(network_params);
    } else {
        train(network_params);
    }
}
