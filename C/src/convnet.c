#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "matrix_type.h"
#include "matrix_operations.h"
#include "convnet_operations.h"
#include "convnet_layers.h"
#include "convnet.h"

int trainConvnet(ConvnetParameters* network_params) {
	ThreeDMatrix** training_data = network_params->X;
	
}