#ifndef __CONVNET_HEADER__
#define __CONVNET_HEADER__

typedef struct {
	ThreeDMatrix** X;
	ThreeDMatrix** correct_labels;
	
};

int trainConvnet(ConvnetParameters* network_params);
#endif