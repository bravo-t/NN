#include "matrix_operations.h"
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#include "layers.h"
#include "fully_connected_net.h"

int fullyConnectedNets(TwoDMatrix* X, int minibatch_size) {

}

int initNetwork() {
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
}