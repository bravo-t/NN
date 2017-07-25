#include "matrix_operations.h"
#include <stdlib.h>
#include "misc_utils.h"
#include <math.h>
#include "layers.h"
#include "fully_connected_net.h"
#include <malloc.h>
#include <stdarg.h>

int fullyConnectedNets(TwoDMatrix* X, int minibatch_size) {

}

int initParameters(TwoDMatrix* X, 
	TwoDMatrix* correct_labels, 
	int minibatch_size, 
	int labels, 
	float learning_rate, 
	float reg_strength, 
	float alpha, 
	int network_depth, ...) {
	va_list size_configs;
	int layer_sizes[network_depth];
	va_start(size_configs,network_depth);
	for(int i=0;i<network_depth;i++) {
		layer_sizes[i] = va_arg(size_configs,int);
	}
	va_end(size_configs);
	// The last layer is the label layer, so you don't have control on the size of it
	layer_sizes[network_depth-1] = labels;
}

	/*
	 How will the size of the Ws and Hs determined?
	 Like below (X is the input data):
	 int former_width = X->width;
	 TwoDMatrix** Ws;
	 TwoDMatrix** bs;
	 for(int i=0;i<network_depth;i++) {
		init2DMatrix(Ws[i],former_width,hidden_layer_sizes[i]);
		init2DMatrix(bs[i],1,hidden_layer_sizes[i]);
		former_width = hidden_layer_sizes[i];
	 }
	 */
int train(parameters* networkParams) {
	
}