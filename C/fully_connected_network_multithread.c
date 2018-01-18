#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <stdbool.h>
#include <string.h>
#include "thread_barrier.h"
#include "thread_control.h"
#include "network_type.h"
#include "matrix_operations.h"
#include "matrix_operations_multithread.h"
#include "layers.h"
#include "layers_multithread.h"
#include "misc_utils.h"
#include "fully_connected_net.h"
#include "fully_connected_net_multithread.h"

int main(int argc, char** argv) {
    if (argc == 1) {
        printf("Please specify config file\n");
        exit(0);
    }
    FCParameters* network_params = readNetworkConfigFile(argv[1]);
    int number_of_threads = strtol(argv[2],NULL,10);
    if (strcmp(network_params->mode,"train")) {
        TwoDMatrix* scores = NULL;
        test_multithread(network_params,scores,number_of_threads);
    } else {
        network_params->number_of_threads = number_of_threads;
        train_multithread(network_params);
    }
}
