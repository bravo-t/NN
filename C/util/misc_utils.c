#include <malloc.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "matrix_type.h"
#include "fully_connected_net.h"
#include "misc_utils.h"

TwoDMatrix* matrixMalloc(int size) {
    TwoDMatrix* M = malloc(size);
    memset(M,0,size);
    M->initialized = false;
    return M;
}

void write2DMatrix(FILE* fp, TwoDMatrix* M) {
    for(int i=0;i<M->height;i++) {
        for(int j=0;j<M->width;j++) {
            fprintf(fp,"%f ",M->d[i][j]);
        }
        fprintf(fp,"\n");
    }
}

void getKeyValueFromFile(fp, char** retval) {
    char line[200];
    char delim[] = " =";
    fgets(line, 200, fp);
    char* token=strsep(&line,delim);
    while (token[0] == '\0') {
        token=strsep(&line,delim);
    }
    strcpy(retval[0],token);
    while (token[0] == '\0') {
        token=strsep(&line,delim);
    }
    strcpy(retval[1],line);
}

TwoDMatrix* load2DMatrix(FILE* fp) {
    char buff[8192];
    fscanf(fp,"%s",buff);
    int height,width;
    fscanf(fp,"%d",&height);
    fscanf(fp,"%d",&width);
    float value;
    TwoDMatrix* M = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(M,height,width);
    for(int i=0;i<height;i++) {
        for(int j=0;j<width;j++) {
            fscanf(fp,"%f",&value);
            M->d[i][j] = value;
        }
    }
    return M;
}


int dumpNetworkConfig(int network_depth, int alpha, TwoDMatrix** Ws, TwoDMatrix** bs, bool use_batchnorm, TwoDMatrix** mean_caches, TwoDMatrix** var_caches, TwoDMatrix** gammas, TwoDMatrix** betas, float eps, char* output_dir) {
    int file_name_length = strlen(output_dir) + strlen("/network.params") + 10;
    char* out_file = malloc(sizeof(char)*file_name_length);
    strcpy(out_file,output_dir);
    strcat(out_file,"/network.params");
    printf("INFO: Network parameters dumped to %s\n",out_file);
    
    FILE* out = fopen(out_file,"w");
    if (out == NULL) {
        printf("ERROR: Cannot open %s to read\n",out_file);
        exit(1);
    }
    fprintf(out, "network_depth=%d\n",network_depth);
    fprintf(out, "alpha=%f\n",alpha);
    fprintf(out, "use_batchnorm=%d\n",use_batchnorm);
    if (use_batchnorm) {
        fprintf(out, "batchnorm_eps=%f\n",eps);
    }
    foreach(int i=0;i<network_depth;i++) {
        fprintf(out,"%s[%d] %d %d","W",i,W[i]->height,W[i]->width);
        write2DMatrix(out, W[i]);
    }
    foreach(int i=0;i<network_depth;i++) {
        fprintf(out,"%s[%d] %d %d","b",i,b[i]->height,b[i]->width);
        write2DMatrix(out, b[i]);
    }
    if (use_batchnorm) {
        foreach(int i=0;i<network_depth;i++) {
            fprintf(out,"%s[%d] %d %d","mean_caches",i,mean_caches[i]->height,mean_caches[i]->width);
            write2DMatrix(out, mean_caches[i]);
        }
        foreach(int i=0;i<network_depth;i++) {
            fprintf(out,"%s[%d] %d %d","var_caches",i,var_caches[i]->height,var_caches[i]->width);
            write2DMatrix(out, var_caches[i]);
        }
        foreach(int i=0;i<network_depth;i++) {
            fprintf(out,"%s[%d] %d %d","gammas",i,gammas[i]->height,gammas[i]->width);
            write2DMatrix(out, gammas[i]);
        }
        foreach(int i=0;i<network_depth;i++) {
            fprintf(out,"%s[%d] %d %d","betas",i,betas[i]->height,betas[i]->width);
            write2DMatrix(out, betas[i]);
        }
    }
    fclose(out);
    free(out_file);
    return 0;
}

int loadNetworkConfig(char* dir, int* network_depth, int* alpha, TwoDMatrix** Ws, TwoDMatrix** bs, bool* use_batchnorm, TwoDMatrix** mean_caches, TwoDMatrix** var_caches, TwoDMatrix** gammas, TwoDMatrix** betas, float* batchnorm_eps) {
    int file_name_length = strlen(dir) + strlen("/network.params") + 10;
    char* filename = malloc(sizeof(char)*file_name_length);
    strcpy(filename,dir);
    strcat(filename,"/network.params");
    printf("INFO: Loading network parameters dumped from %s\n",filename);
    FILE* fp = fopen(filename,"r");
    if (fp == NULL) {
        printf("ERROR: Cannot open %s to read\n",filename);
        exit(1);
    }
    char key_values[2][100];
    int i;
    for(i=0;i<3;i++) {
        getKeyValueFromFile(fp,key_values);
        if (! strcmp(key_values[0],"network_depth")) {
            *network_depth = strtol(key_values[1],NULL,10);
        } else if (! strcmp(key_values[0],"alpha")) {
            *alpha = strtof(key_values[1],NULL);
        }  else if (! strcmp(key_values[0],"use_batchnorm")) {
            *use_batchnorm = strtol(key_values[1],NULL,10);
        } else {
            printf("ERROR: Unrecognized keyword: %s, ignored\n",key_values[0]);
        }
    }
    if (use_batchnorm) {
        getKeyValueFromFile(fp,key_values);
        if (! strcmp(key_values[0],"batchnorm_eps")) {
            *batchnorm_eps = strtof(key_values[1],NULL);
        } else {
            printf("ERROR: Unrecognized keyword: %s, ignored\n",key_values[0]);
        }
    }
    
    TwoDMatrix** Ws = malloc(sizeof(TwoDMatrix*)*network_depth);
    TwoDMatrix** bs = malloc(sizeof(TwoDMatrix*)*network_depth);
    foreach(int i=0;i<network_depth;i++) {
        Ws[i] = load2DMatrix(fp);
    }
    foreach(int i=0;i<network_depth;i++) {
        bs[i] = load2DMatrix(fp);
    }
    if (use_batchnorm) {
        gammas = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
        betas = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
        mean_caches = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
        var_caches = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*network_depth);
        foreach(int i=0;i<network_depth;i++) {
            gammas[i] = load2DMatrix(fp);
        }
        foreach(int i=0;i<network_depth;i++) {
            betas[i] = load2DMatrix(fp);
        }
        foreach(int i=0;i<network_depth;i++) {
            mean_caches[i] = load2DMatrix(fp);
        }
        foreach(int i=0;i<network_depth;i++) {
            var_caches[i] = load2DMatrix(fp);
        }
    }
    fclose(fp);
    return 0;
}

parameters* readNetworkConfigFile(char* filename) {
    FILE* fp = fopen(filename,"r");
    if (fp == NULL) {
        printf("ERROR: Cannot open %s to read\n",filename);
        exit(1);
    }
    parameters* network_params = malloc(sizeof(parameters));
    // Assign default values 
    
    char key_values[2][8192];
    bool network_depth_defined = false;
    while (! feof(fp)) {
        getKeyValueFromFile(fp,key_values);
        if (! strcmp(key_values[0],"training_data")) {
            network_params->X = load2DMatrixFromFile(key_values[1]);
        } else if (! strcmp(key_values[0],"correct_labels")) {
            network_params->correct_labels = load2DMatrixFromFile(key_values[1]);
        } else if (! strcmp(key_values[0],"hidden_layer_sizes")) {
            int layers[8192];
            int network_depth = 0;
            for(token = strsep(&key_values[1], " "); token != NULL; token = strsep(&key_values[1], " ")) {
                if (token[0] != '\0') {
                    network_depth++;
                    layers[network_depth] = strtol(token,NULL,10);;
                }
            }
            network_params->hidden_layer_sizes = (int*) malloc(sizeof(int)*network_depth);
            for(int i=0;i<network_depth;i++) {
                network_params->hidden_layer_sizes[i] = layers[i];
            }
            network_params->network_depth = network_depth;
        } else if (! strcmp(key_values[0],"")) {

        } else if (! strcmp(key_values[0],"")) {

        } else if (! strcmp(key_values[0],"")) {

        } else if (! strcmp(key_values[0],"")) {

        } else if (! strcmp(key_values[0],"")) {

        } else if (! strcmp(key_values[0],"")) {

        } else if (! strcmp(key_values[0],"")) {

        } else 
    }
}
