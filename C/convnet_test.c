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

int readCIFARDataFile(char* filename, int start_index, ThreeDMatrix** X, TwoDMatrix* correct_labels);
int readCIFARData(char* dir, int file_number, ThreeDMatrix** X, TwoDMatrix* correct_labels);

int main() {
	ThreeDMatrix** X = malloc(sizeof(ThreeDMatrix*)*50000);
	for(int i=0;i<50000;i++) {
		X[i] = matrixMalloc(sizeof(ThreeDMatrix));
	}
	TwoDMatrix* correct_labels = matrixMalloc(sizeof(TwoDMatrix));
	init2DMatrix(correct_labels, 50000, 1);
	readCIFARData("/cygdrive/d/cifar-10-binary/cifar-10-batches-bin",
		5,
		X,
		correct_labels);
	return 0;
}

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
		strcat(filepath, "/");
		strcat(filepath, filenames[i]);
		printf("INFO: Reading %s\n", filepath);
		readCIFARDataFile(filepath, i*10000, X, correct_labels);
	}
	char** category = malloc(sizeof(char*)*10);
	category[0] = "airplane";
	category[1] = "automobile";
	category[2] = "bird";
	category[3] = "cat";
	category[4] = "deer";
	category[5] = "dog";
	category[6] = "frog";
	category[7] = "horse";
	category[8] = "ship";
	category[9] = "truck";
	int count[10] = {0};
	for(int i=0;i<correct_labels->height;i++) {
		printf("%f\n",correct_labels->d[i][0]);
		int index = (int) correct_labels->d[i][0];
		count[index]++;
	}
	for(int i=0;i<10;i++) {
		if (count[i] > 0) {
			printf("INFO: %d training samples read for category %s\n",count[i],category[i]);
		}
	}
	free(category);
	free(filenames);
	return 0;
}