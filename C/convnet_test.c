#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <malloc.h>
#include <string.h>
#include "network_type.h"
#include "matrix_operations.h"
#include "layers.h"
#include "misc_utils.h"
#include "fully_connected_net.h"
#include "convnet_operations.h"
#include "convnet_layers.h"
#include "convnet.h"

int readCIFARDataFile(char* filename, int start_index, ThreeDMatrix** X, TwoDMatrix* correct_labels) {
	for(int i=start_index;i<(start_index+10000);i++) {
		init3DMatrix(X[i], 3, 32, 32);
	}
	if (! (correct_labels->initialized)) {
		printf("ERROR: 2D matrix must be initialized before use\n");
		exit(1);
	}
	FILE *fp = fopen(filename, "rb");
	if (fp == NULL) {
		printf("ERROR: Cannot open file %s\n", filename);
		exit(1);
	}
	unsigned char buffer[3073];
	for(int i=0;i<10000;i++) {
		fread(buffer, sizeof(buffer),1,fp);
		int actual_index = start_index + i;
		correct_labels->d[actual_index][1] = buffer[0];
		for(int m=0;m<3;m++) {
			for(int n=0;n<32;n++) {
				for(int o=0;o<32;o++) {
					int data_index = m*n*o + 1;
					X[actual_index]->d[m][n][o] = buffer[data_index];
				}
			}
		}
	}
	fclose(fp);
	return 0;
}

int readCIFARData(char* dir, int file_number, ThreeDMatrix** X, TwoDMatrix* correct_labels) {
	char** filenames = malloc(sizeof(char*)*5);
	filenames[0] = "data_batch_1.bin";
	filenames[1] = "data_batch_2.bin";
	filenames[2] = "data_batch_3.bin";
	filenames[3] = "data_batch_4.bin";
	filenames[4] = "data_batch_5.bin";
	init2DMatrix(correct_labels, 10000*file_number, 1);
	for(int i=0;i<file_number;i++) {
		int name_length = strlen(filenames[i]) + strlen(dir) + 5;
		char* filepath = malloc(sizeof(char) * name_length);
		strcpy(filepath, dir);
		strcpy(filepath, "/");
		strcpy(filepath, filenames[i]);
		readCIFARDataFile(filepath, i*10000, X, correct_labels);
	}
	
}