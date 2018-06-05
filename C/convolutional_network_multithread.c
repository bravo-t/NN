#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <malloc.h>
#include <math.h>
#include <string.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>
#include "src/thread_barrier.h"
#include "src/thread_control.h"
#include "src/network_type.h"
#include "src/matrix_operations.h"
#include "src/matrix_operations_multithread.h"
#include "src/layers.h"
#include "src/layers_multithread.h"
#include "src/misc_utils.h"
#include "src/fully_connected_net.h"
#include "src/fully_connected_net_multithread.h"
#include "src/convnet_operations.h"
#include "src/convnet_layers.h"
#include "src/convnet.h"
#include "src/convnet_multithread.h"

int main(int argc, char** argv) {
    if (argc == 1) {
        printf("Please specify config file\n");
        exit(0);
    }
    ConvnetParameters* network_params = readConvnetConfigFile(argv[1]);
    int number_of_threads = strtol(argv[2],NULL,10);
    if (strcmp(network_params->mode,"train")) {
    	TwoDMatrix* scores = NULL;
        testConvnet_multithread(network_params,scores);
    } else {
        network_params->number_of_threads = number_of_threads;
        trainConvnet_multithread(network_params);
    }
}
