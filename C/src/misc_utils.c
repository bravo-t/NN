#include <stdio.h>
#include <malloc.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "network_type.h"
#include "matrix_operations.h"
#include "fully_connected_net.h"
#include "misc_utils.h"

TwoDMatrix* load2DMatrixFromFile(char* filename) {
    FILE* fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("ERROR: Cannot open %s to read\n",filename);
        exit(1);
    }
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
    fclose(fp);
    return M;
}

float matrixError(TwoDMatrix* a, TwoDMatrix* b) {
    if (a->height != b->height) {
        printf("HOLY ERROR: Height does not match, your code is really messed up\n");
        return 1.0/0.0;
    }
    if (a->width != b->width) {
        printf("ANOTHER ERROR: Width doesn't match. FIX THEM\n");
        return 1.0/0.0;
    }
    TwoDMatrix* sub = matrixMalloc(sizeof(TwoDMatrix));
    init2DMatrix(sub, a->height, a->width);
    elementwiseSub2DMatrix(a, b, sub);
    float error = 0;
    for(int i=0;i<sub->height;i++) {
        for(int j=0;j<sub->width;j++) {
            if (sub->d[i][j] > 0) {
                error += sub->d[i][j];
            } else {
                error -= sub->d[i][j];
            }
        }
    }
    destroy2DMatrix(sub);
    return error;
}

void printMatrix(TwoDMatrix *M) {
    printf("Height of matrix: %d, width: %d\n",M->height,M->width);
    for(int i=0;i<M->height;i++) {
        for(int j=0;j<M->width;j++) {
            printf("%f\t",M->d[i][j]);
        }
        printf("\n");
    }
}

void __debugPrintMatrix(TwoDMatrix *M, char* M_name) {
    printf("%s = \n",M_name);
    printf("Height of matrix: %d, width: %d\n",M->height,M->width);
    for(int i=0;i<M->height;i++) {
        for(int j=0;j<M->width;j++) {
            printf("%f\t",M->d[i][j]);
        }
        printf("\n");
    }
}


void checkMatrixDiff(TwoDMatrix* a, TwoDMatrix* b, float thres) {
    float diff = matrixError(a, b);
    if (diff >= thres) {
        printf("ERROR: Difference between ref and impl is too big: %f\n",diff);
        printf("ref = \n");
        printMatrix(a);
        printf("impl = \n");
        printMatrix(b);
    } else {
        printf("Difference of the two matrixes are %f\n",diff);
    }
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

void write2DMatrix(FILE* fp, TwoDMatrix* M) {
    for(int i=0;i<M->height;i++) {
        for(int j=0;j<M->width;j++) {
            fprintf(fp,"%f ",M->d[i][j]);
        }
        fprintf(fp,"\n");
    }
}

void getKeyValueFromFile(FILE* fp, char** retval) {
    char* line = malloc(sizeof(char)*200);
    char* line_start = line;
    char delim[] = " =";
    fgets(line, 200, fp);
    if (strlen(line) <= 2) {
        // This is a empty line
        retval[0][0] = '\0';
        retval[1][0] = '\0';
        free(line_start);
        return;
    }
    for(int i=0;i<200;i++) {
        if (line[i] == '\n' || line[i] == '\r') {
            line[i] = '\0';
            break;
        }
    }
    char* token=strsep(&line,delim);
    //printf("DEBUG: token=%s,line=%s\n",token,line);
    while ((token[0] == '\0' || token[0] == ' ' || token[0] == '=') && line != NULL) {
        token=strsep(&line,delim);
        //printf("DEBUG: token=%s,line=%s\n",token,line);
    }
    strcpy(retval[0],token);
    token=strsep(&line,delim);
    //printf("DEBUG: token=%s,line=%s\n",token,line);
    while ((token[0] == '\0' || token[0] == ' ' || token[0] == '=') && line != NULL) {
        token=strsep(&line,delim);
        //printf("DEBUG: token=%s,line=%s\n",token,line);
    }
    if (line != NULL) {
        strcpy(retval[1],strcat(token,line));
    } else {
        strcpy(retval[1],token);
    }
    free(line_start);
}



int dumpNetworkConfig(int network_depth, float alpha, TwoDMatrix** Ws, TwoDMatrix** bs, bool use_batchnorm, TwoDMatrix** mean_caches, TwoDMatrix** var_caches, TwoDMatrix** gammas, TwoDMatrix** betas, float eps, char* output_dir) {
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
    for(int i=0;i<network_depth;i++) {
        fprintf(out,"%s[%d] %d %d\n","Ws",i,Ws[i]->height,Ws[i]->width);
        write2DMatrix(out, Ws[i]);
    }
    for(int i=0;i<network_depth;i++) {
        fprintf(out,"%s[%d] %d %d\n","bs",i,bs[i]->height,bs[i]->width);
        write2DMatrix(out, bs[i]);
    }
    if (use_batchnorm) {
        for(int i=0;i<network_depth;i++) {
            fprintf(out,"%s[%d] %d %d\n","mean_caches",i,mean_caches[i]->height,mean_caches[i]->width);
            write2DMatrix(out, mean_caches[i]);
        }
        for(int i=0;i<network_depth;i++) {
            fprintf(out,"%s[%d] %d %d\n","var_caches",i,var_caches[i]->height,var_caches[i]->width);
            write2DMatrix(out, var_caches[i]);
        }
        for(int i=0;i<network_depth;i++) {
            fprintf(out,"%s[%d] %d %d\n","gammas",i,gammas[i]->height,gammas[i]->width);
            write2DMatrix(out, gammas[i]);
        }
        for(int i=0;i<network_depth;i++) {
            fprintf(out,"%s[%d] %d %d\n","betas",i,betas[i]->height,betas[i]->width);
            write2DMatrix(out, betas[i]);
        }
    }
    fclose(out);
    free(out_file);
    return 0;
}

int loadNetworkConfig(char* dir, int* network_depth, float* alpha, TwoDMatrix*** Ws, TwoDMatrix*** bs, bool* use_batchnorm, TwoDMatrix*** mean_caches, TwoDMatrix*** var_caches, TwoDMatrix*** gammas, TwoDMatrix*** betas, float* batchnorm_eps) {
    int file_name_length = strlen(dir) + strlen("/network.params") + 10;
    char* filename = malloc(sizeof(char)*file_name_length);
    strcpy(filename,dir);
    strcat(filename,"/network.params");
    printf("INFO: Loading network parameters from %s\n",filename);
    FILE* fp = fopen(filename,"r");
    if (fp == NULL) {
        printf("ERROR: Cannot open %s to read\n",filename);
        exit(1);
    }
    char** key_values = malloc(sizeof(char*)*2);
    key_values[0] = (char*) malloc(sizeof(char)*100);
    key_values[1] = (char*) malloc(sizeof(char)*100);
    for(int i=0;i<3;i++) {
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
    if (*use_batchnorm) {
        getKeyValueFromFile(fp,key_values);
        if (! strcmp(key_values[0],"batchnorm_eps")) {
            *batchnorm_eps = strtof(key_values[1],NULL);
        } else {
            printf("ERROR: Unrecognized keyword: %s, ignored\n",key_values[0]);
        }
    }
    
    *Ws = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*(*network_depth));
    *bs = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*(*network_depth));
    for(int i=0;i<*network_depth;i++) {
        (*Ws)[i] = load2DMatrix(fp);
    }
    for(int i=0;i<*network_depth;i++) {
        (*bs)[i] = load2DMatrix(fp);
    }
    if (use_batchnorm) {
        *gammas = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*(*network_depth));
        *betas = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*(*network_depth));
        *mean_caches = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*(*network_depth));
        *var_caches = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*(*network_depth));
        for(int i=0;i<*network_depth;i++) {
            (*gammas)[i] = load2DMatrix(fp);
        }
        for(int i=0;i<*network_depth;i++) {
            (*betas)[i] = load2DMatrix(fp);
        }
        for(int i=0;i<*network_depth;i++) {
            (*mean_caches)[i] = load2DMatrix(fp);
        }
        for(int i=0;i<*network_depth;i++) {
            (*var_caches)[i] = load2DMatrix(fp);
        }
    }
    fclose(fp);
    free(key_values[0]);
    free(key_values[1]);
    free(key_values);
    return 0;
}


FCParameters* readNetworkConfigFile(char* filename) {
    FILE* fp = fopen(filename,"r");
    if (fp == NULL) {
        printf("ERROR: Cannot open %s to read\n",filename);
        exit(1);
    }
    FCParameters* network_params = malloc(sizeof(FCParameters));
    // Assign default values 
    network_params->reg_strength = 1e-2;
    network_params->learning_rate = 0.01f;
    network_params->alpha = 0.0f;
    network_params->verbose = false;
    network_params->use_momentum_update = false;
    network_params->use_nag_update = false;
    network_params->use_rmsprop = false;
    network_params->mu = 0.5f;
    network_params->decay_rate = 0.99f;
    network_params->eps = 1e-6;
    network_params->use_batchnorm = false;
    network_params->batchnorm_momentum = 0.5f;
    network_params->batchnorm_eps = 1e-6;

    char** key_values = malloc(sizeof(char*)*2);
    key_values[0] = (char*) malloc(sizeof(char)*8192);
    key_values[1] = (char*) malloc(sizeof(char)*8192);
    bool mode_defined = false;
    bool params_dir_defined = false;
    bool data_set_defined = false;
    bool correct_labels_defined = false;
    bool hidden_layer_sizes_defined = false;
    bool labels_defined = false;
    bool epochs_defined = false;
    bool minibatch_defined = false;
    while (! feof(fp)) {
        key_values[0][0] = '\0';
        key_values[1][0] = '\0';
        getKeyValueFromFile(fp,key_values);
        if (key_values[0][0] == '#' || key_values[0][0] == '\0') {
            continue;
        }
        if (! strcmp(key_values[0],"data_set")) {
            network_params->X = load2DMatrixFromFile(key_values[1]);
            data_set_defined = true;
        } else if (! strcmp(key_values[0],"correct_labels")) {
            network_params->correct_labels = load2DMatrixFromFile(key_values[1]);
            correct_labels_defined = true;
        } else if (! strcmp(key_values[0],"hidden_layer_sizes")) {
            int layers[8192];
            int network_depth = 0;
            char* sizes = malloc(sizeof(char)*8192);
            strcpy(sizes,key_values[1]);
            char* sizes_ptr = sizes;
            for(char* token = strsep(&sizes_ptr, ","); token != NULL; token = strsep(&sizes_ptr, ",")) {
                if (token[0] != '\0') {
                    layers[network_depth] = strtol(token,NULL,10);
                    network_depth++;
                }
            }
            network_params->hidden_layer_sizes = (int*) malloc(sizeof(int)*network_depth);
            for(int i=0;i<network_depth;i++) {
                network_params->hidden_layer_sizes[i] = layers[i];
            }
            network_params->network_depth = network_depth;
            hidden_layer_sizes_defined = true;
            free(sizes);
        } else if (! strcmp(key_values[0],"labels")) {
            network_params->labels = strtol(key_values[1],NULL,10);
            labels_defined = true;
        } else if (! strcmp(key_values[0],"minibatch_size")) {
            network_params->minibatch_size = strtol(key_values[1],NULL,10);
            minibatch_defined = true;
        } else if (! strcmp(key_values[0],"alpha")) {
            network_params->alpha = strtof(key_values[1],NULL);
        } else if (! strcmp(key_values[0],"learing_rate")) {
            network_params->learning_rate = strtof(key_values[1],NULL);
        } else if (! strcmp(key_values[0],"epochs")) {
            network_params->epochs = strtol(key_values[1],NULL,10);
            epochs_defined = true;
        } else if (! strcmp(key_values[0],"verbose")) {
            network_params->verbose = strtol(key_values[1],NULL,10);
        } else if (! strcmp(key_values[0],"use_momentum_update")) {
            network_params->use_momentum_update = strtol(key_values[1],NULL,10);
        } else if (! strcmp(key_values[0],"use_nag_update")) {
            network_params->use_nag_update = strtol(key_values[1],NULL,10);
        } else if (! strcmp(key_values[0],"use_rmsprop")) {
            network_params->use_rmsprop = strtol(key_values[1],NULL,10);
        } else if (! strcmp(key_values[0],"mu")) {
            network_params->mu = strtof(key_values[1],NULL);
        } else if (! strcmp(key_values[0],"decay_rate")) {
            network_params->decay_rate = strtof(key_values[1],NULL);
        } else if (! strcmp(key_values[0],"eps")) {
            network_params->eps = strtof(key_values[1],NULL);
        } else if (! strcmp(key_values[0],"use_batchnorm")) {
            network_params->use_batchnorm = strtol(key_values[1],NULL,10);
        } else if (! strcmp(key_values[0],"batchnorm_momentum")) {
            network_params->batchnorm_momentum = strtof(key_values[1],NULL);
        } else if (! strcmp(key_values[0],"batchnorm_eps")) {
            network_params->batchnorm_eps = strtof(key_values[1],NULL);
        } else if (! strcmp(key_values[0],"mode")) {
            network_params->mode = (char*) malloc(sizeof(char)*strlen(key_values[1]));
            strcpy(network_params->mode,key_values[1]);
            mode_defined = true;
        } else if (! strcmp(key_values[0],"params_dir")) {
            network_params->params_save_dir = (char*) malloc(sizeof(char)*strlen(key_values[1]));
            strcpy(network_params->params_save_dir,key_values[1]);
            params_dir_defined = true;
        } else {
            printf("ERROR: Unrecognized keyword %s, ignored\n",key_values[0]);
        }
    }
    free(key_values[0]);
    free(key_values[1]);
    free(key_values);
    if (! mode_defined) {
        printf("ERROR: Mode not defined\n");
        exit(1);
    }
    if (! params_dir_defined) {
        printf("ERROR: Dir to load or save params not defined\n");
        exit(1);
    }
    if (! data_set_defined) {
        printf("ERROR: Data set not defined\n");
        exit(1);
    }
    if (! correct_labels_defined && strcmp(network_params->mode,"test")) {
        printf("ERROR: Correct labels not defined\n");
        exit(1);
    }
    if (! hidden_layer_sizes_defined && strcmp(network_params->mode,"test")) {
        printf("ERROR: Sizes of hidden layers not defined\n");
        exit(1);
    }
    if (! minibatch_defined) {
        printf("ERROR: Minibatch size not defined\n");
        exit(1);
    }
    if (! labels_defined && strcmp(network_params->mode,"test")) {
        printf("ERROR: Number of lables not defined\n");
        exit(1);
    }
    if (! epochs_defined && strcmp(network_params->mode,"test")) {
        printf("ERROR: Epochs not defined\n");
        exit(1);
    }
    network_params->hidden_layer_sizes[network_params->network_depth-1] = network_params->labels;
    return network_params;
}

char determineMemoryUnit(unsigned int n) {
    char units[] = {" ","K","M","G","T"};
    float n_float = (float) n;
    for(int i=0;i<5;i++) {
        if (n_float < 1) return units[i-1];
        n_float = n_float/1024;
    }
    return units[0];
}

float memoryUsageReadable(unsigned int n, char unit) {
    switch (unit) {
        case " ": return (float) n;
        case "K": return (float) n/1024;
        case "M": return (float) n/(1024*1024);
        case "M": return (float) n/(1024*1024*1024);
        case "M": return (float) n/(1024*1024*1024*1024);
        default: return (float) n;
    }
}
