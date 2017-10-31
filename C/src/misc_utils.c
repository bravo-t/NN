#include <stdio.h>
#include <malloc.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
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

ThreeDMatrix* load3DMatrixFromFile(char* filename) {
    FILE* fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("ERROR: Cannot open %s to read\n",filename);
        exit(1);
    }
    char buff[8192];
    fscanf(fp,"%s",buff);
    int depth,height,width;
    fscanf(fp,"%d",&depth);
    fscanf(fp,"%d",&height);
    fscanf(fp,"%d",&width);
    float value;
    ThreeDMatrix* M = matrixMalloc(sizeof(ThreeDMatrix));
    init3DMatrix(M,depth,height,width);
    for(int i=0;i<depth;i++) {
        for(int j=0;j<height;j++) {
            for(int k=0;k<width;k++) {
                fscanf(fp,"%f",&value);
                M->d[i][j][k] = value;
            }
        }
    }
    fclose(fp);
    return M;
}

ThreeDMatrix** load3DMatricesFromFile(char* filename, int number_of_matrices) {
    FILE* fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("ERROR: Cannot open %s to read\n",filename);
        exit(1);
    }
    ThreeDMatrix** M = malloc(sizeof(ThreeDMatrix*)*number_of_matrices);
    for(int i=0;i<number_of_matrices;i++) {
        char buff[8192];
        fscanf(fp,"%s",buff);
        int depth,height,width;
        fscanf(fp,"%d",&depth);
        fscanf(fp,"%d",&height);
        fscanf(fp,"%d",&width);
        float value;
        M[i] = (ThreeDMatrix*) matrixMalloc(sizeof(ThreeDMatrix));
        init3DMatrix(M[i],depth,height,width);
        for(int z=0;z<depth;z++) {
            for(int y=0;y<height;y++) {
                for(int x=0;x<width;x++) {
                    fscanf(fp,"%f",&value);
                    M[i]->d[z][y][x] = value;
                }
            }
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

float matrixError3D(ThreeDMatrix* a, ThreeDMatrix* b) {
    if (a->height != b->height) {
        printf("HOLY ERROR: Height does not match, your code is really messed up\n");
        return 1.0/0.0;
    }
    if (a->width != b->width) {
        printf("ANOTHER ERROR: Width doesn't match. FIX THEM\n");
        return 1.0/0.0;
    }
    if (a->depth != b->depth) {
        printf("ERROR AGAIN: Depth not equal, oh god\n");
    }
    float error = 0;
    for(int i=0;i<a->depth;i++) {
        for(int j=0;j<a->height;j++) {
            for(int k=0;k<a->width;k++) {
                float sub = a->d[i][j][k] - b->d[i][j][k];
                if (sub > 0) {
                    error += sub;
                } else {
                    error -= sub;
                }
            }
        }
    }
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

void print3DMatrix(ThreeDMatrix *M) {
    printf("Depth of matrix: %d, height: %d, width: %d\n",M->depth,M->height,M->width);
    for(int i=0;i<M->depth;i++) {
        for(int j=0;j<M->height;j++) {
            for(int k=0;k<M->width;k++) {
                printf("%f\t",M->d[i][j][k]);
            }
            printf("\n");
        }
        printf("--------------------------\n");
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
        printf("Difference of the two matrices are %f\n",diff);
    }
}

void check3DMatrixDiff(ThreeDMatrix* a, ThreeDMatrix* b, float thres) {
    float diff = matrixError3D(a, b);
    if (diff >= thres) {
        printf("ERROR: Difference between ref and impl is too big: %f\n",diff);
        printf("ref = \n");
        print3DMatrix(a);
        printf("impl = \n");
        print3DMatrix(b);
    } else {
        printf("Difference of the two matrices are %f\n",diff);
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

ThreeDMatrix* load3DMatrix(FILE* fp) {
    char buff[8192];
    fscanf(fp,"%s",buff);
    int depth,height,width;
    fscanf(fp,"%d",&depth);
    fscanf(fp,"%d",&height);
    fscanf(fp,"%d",&width);
    float value;
    ThreeDMatrix* M = matrixMalloc(sizeof(ThreeDMatrix));
    init3DMatrix(M,depth,height,width);
    for(int i=0;i<depth;i++) {
        for(int j=0;j<height;j++) {
            for(int k=0;k<width;k++) {
                fscanf(fp,"%f",&value);
                M->d[i][j][k] = value;
            }
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

void write3DMatrix(FILE* fp, ThreeDMatrix* M) {
    for(int i=0;i<M->depth;i++) {
        for(int j=0;j<M->height;j++) {
            for(int k=0;k<M->width;k++) {
                fprintf(fp,"%f ",M->d[i][j][k]);
            }
            fprintf(fp,"\n");
        }
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
    network_params->enable_learning_rate_step_decay = false;
    network_params->enable_learning_rate_exponential_decay = false;
    network_params->enable_learning_rate_invert_t_decay = false;


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
        } else if (! strcmp(key_values[0],"shuffle_training_samples")) {
            network_params->shuffle_training_samples = strtof(key_values[1],NULL);
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
    char units[5] = {' ','K','M','G','T'};
    float n_float = (float) n;
    for(int i=0;i<5;i++) {
        if (n_float < 1) return units[i-1];
        n_float = n_float/1024;
    }
    return units[0];
}

float memoryUsageReadable(unsigned long long int n, char unit) {
    switch (unit) {
        case ' ': return (float) n;
        case 'K': return (float) (n>>10);
        case 'M': return (float) (n>>20);
        case 'G': return (float) (n>>30);
        case 'T': return (float) (n>>40);
        default: return (float) n;
    }
}

int writeImage(ThreeDMatrix* X, char* var_name, char* img_dir) {
    ThreeDMatrix* X_normalized = matrixMalloc(sizeof(ThreeDMatrix));
    init3DMatrix(X_normalized,X->depth,X->height,X->width);
    for(int i=0;i<X_normalized->depth;i++) {
        float min_value = X->d[i][0][0];
        float max_value = X->d[i][0][0];
        for(int j=0;j<X->height;j++) {
            for(int k=0;k<X->width;k++) {
                if (X->d[i][j][k] > max_value) max_value = X->d[i][j][k];
                if (X->d[i][j][k] < min_value) min_value = X->d[i][j][k];
            }
        }
        float scale_factor = 255/(max_value - min_value);
        for(int j=0;j<X_normalized->height;j++) {
            for(int k=0;k<X_normalized->width;k++) {
                X_normalized->d[i][j][k] = X->d[i][j][k] - min_value;
                X_normalized->d[i][j][k] = X_normalized->d[i][j][k] * scale_factor;
            }
        }
    }
    for(int i=0;i<X_normalized->depth;i++) {
        char* file = malloc(sizeof(char)*(strlen(var_name) + strlen(img_dir) + 20));
        strcpy(file,img_dir);
        strcat(file,"/");
        strcat(file,var_name);
        strcat(file,".");
        char counter[50];
        sprintf(counter,"%d",i);
        strcat(file,counter);
        strcat(file,".pgm");
        FILE *fp = fopen(file, "w");
        if (fp == NULL) {
            printf("ERROR: Cannot open file %s\n", file);
            continue;
        }
        fprintf(fp, "P2\n%d %d\n255\n", X_normalized->width, X_normalized->height);
        for(int j=0;j<X_normalized->height;j++) {
            for(int k=0;k<X_normalized->width;k++) {
                fprintf(fp, "%d ", ((int) X_normalized->d[i][j][k]));
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
        free(file);
    }
    if (X_normalized->depth == 3) {
        char* file = malloc(sizeof(char)*(strlen(var_name) + strlen(img_dir) + 20));
        strcpy(file,img_dir);
        strcat(file,"/");
        strcat(file,var_name);
        strcat(file,".ppm");
        FILE *fp = fopen(file, "w");
        if (fp == NULL) {
            printf("ERROR: Cannot open file %s\n", file);
            return 1;
        }
        fprintf(fp, "P3\n%d %d\n255\n", X_normalized->width, X_normalized->height);
        for(int i=0;i<X_normalized->height;i++) {
            for(int j=0;j<X_normalized->width;j++) {
                for(int k=0;k<X_normalized->depth;k++) {
                    fprintf(fp, "%d ", ((int) X_normalized->d[k][i][j]));
                }
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
        free(file);
    }
    destroy3DMatrix(X_normalized);
    return 0;
}

int verticallyFlipSample(ThreeDMatrix* in, ThreeDMatrix* out) {
    init3DMatrix(out, in->depth, in->height, in->width);
    int half_height = in->height / 2;
    float tmp = 0.0f;
    for(int i=0;i<in->depth;i++) {
        for(int j=0;j<half_height;j++) {
            int swap_index = out->height - 1 - j;
            if ( j != swap_index) {
                for (int k=0;k<in->width;k++) {
                    tmp = in->d[i][j][k];
                    out->d[i][j][k] = in->d[i][swap_index][k];
                    out->d[i][swap_index][k] = tmp;
                }
            }
        }
    }
    return 0;
}

int horizontallyFlipSample(ThreeDMatrix* in, ThreeDMatrix* out) {
    init3DMatrix(out, in->depth, in->height, in->width);
    int half_width = in->width / 2;
    float tmp = 0.0f;
    for(int i=0;i<in->depth;i++) {
        for(int j=0;j<in->height;j++) {
            for (int k=0;k<half_width;k++) {
                int swap_index = out->width - 1 - j;
                if (k != swap_index) {
                    tmp = in->d[i][j][k];
                    out->d[i][j][k] = in->d[i][j][swap_index];
                    out->d[i][j][swap_index] = tmp;
                }
            }
        }
    }
    return 0;
}

int shuffleTrainingSamples(ThreeDMatrix** data_in, 
    TwoDMatrix* label_in,
    int number_of_samples, 
    bool vertically_flip_samples,
    bool horizontally_flip_samples,
    ThreeDMatrix** data_out,
    TwoDMatrix* label_out) {
    if (label_out != label_in) copyTwoDMatrix(label_in, label_out);
    srand(time(NULL));
    if (number_of_samples > 1) {
        for(int i=number_of_samples-1;i>1;i--) {
            int j = i + (int) ((number_of_samples-i) * (rand() / (RAND_MAX + 1.0)));
            ThreeDMatrix* tmp = data_in[i];
            data_out[i] = data_in[j];
            data_out[j] = tmp;
            float label_tmp = label_in->d[i][0];
            label_out->d[i][0] = label_in->d[j][0];
            label_out->d[j][0] = label_tmp;
            if (vertically_flip_samples) {
                if (rand() % 2) {
                    verticallyFlipSample(data_out[i],data_out[i]);
                }
            }
            if (horizontally_flip_samples) {
                if (rand() % 2) {
                    horizontallyFlipSample(data_out[i],data_out[i]);
                }
            }
        }
    } else {
        data_out[0] = data_in[0];
    }
    return 0;
}

int shuffleTrainingSamplesFCNet(TwoDMatrix* data_in, TwoDMatrix* label_in, TwoDMatrix* data_out, TwoDMatrix* label_out) {
    if (data_out != data_in) copyTwoDMatrix(data_in, data_out);
    if (label_out != label_in) copyTwoDMatrix(label_in, label_out);
    int number_of_samples = data_in->height;
    int data_length = sizeof(float)*(data_in->width);
    if (number_of_samples > 1) {
        for(int i=number_of_samples-1;i>1;i--) {
            int j = i + (int) ((number_of_samples-i) * (rand() / (RAND_MAX + 1.0)));
            float* tmp = malloc(data_length);
            memcpy(tmp,data_in->d[i],data_length);
            memcpy(data_out->d[i],data_in->d[j],data_length);
            memcpy(data_out->d[j],tmp,data_length);
            free(tmp);
            float label_tmp = label_in->d[i][0];
            label_out->d[i][0] = label_in->d[j][0];
            label_out->d[j][0] = label_tmp;
        }
    }
    return 0;
}

int splitStr(char* input, int* output) {
    char* str = malloc(sizeof(char)*8192);
    strcpy(str, input);
    char* str_start = str;
    int i = 0;
    for(char* token = strsep(&str, ","); token != NULL; token = strsep(&str, ",")) {
        if (token[0] != '\0') {
            output[i] = strtol(token,NULL,10);
            i++;
        }
    }
    free(str_start);
    return 0;
}

int dumpConvnetConfig(int M,int N,
    int* filter_number,int* filter_stride_x, int* filter_stride_y, int* filter_width, int* filter_height, 
    bool* enable_maxpooling,int* pooling_stride_x,int* pooling_stride_y,int* pooling_width,int* pooling_height,
    int* padding_width, int* padding_height,
    float alpha, bool normalize_data_per_channel, int K,
    ThreeDMatrix**** F,ThreeDMatrix**** b,
    TwoDMatrix** Ws,TwoDMatrix** bs,
    char* output_dir) {
    int file_name_length = strlen(output_dir) + strlen("/convnet.params") + 10;
    char* out_file = malloc(sizeof(char)*file_name_length);
    strcpy(out_file,output_dir);
    strcat(out_file,"/convnet.params");
    
    FILE* out = fopen(out_file,"w");
    if (out == NULL) {
        printf("ERROR: Cannot open %s to read\n",out_file);
        exit(1);
    }
    
    fprintf(out, "M=%d\n",M);
    fprintf(out, "N=%d\n",N);
    fprintf(out, "K=%d\n",K);
    fprintf(out, "alpha=%f\n", alpha);
    fprintf(out, "normalize_data_per_channel=%d\n", normalize_data_per_channel);
    fprintf(out, "filter_number=");
    for(int i=0;i<M*N;i++) {
        fprintf(out, "%d",filter_number[i]);
        if (i != M*N-1) fprintf(out, ",");
    }
    fprintf(out, "\n");
    fprintf(out, "filter_stride_x=");
    for(int i=0;i<M*N;i++) {
        fprintf(out, "%d",filter_stride_x[i]);
        if (i != M*N-1) fprintf(out, ",");
    }
    fprintf(out, "\n");
    fprintf(out, "filter_stride_y=");
    for(int i=0;i<M*N;i++) {
        fprintf(out, "%d",filter_stride_y[i]);
        if (i != M*N-1) fprintf(out, ",");
    }
    fprintf(out, "\n");
    fprintf(out, "filter_width=");
    for(int i=0;i<M*N;i++) {
        fprintf(out, "%d",filter_width[i]);
        if (i != M*N-1) fprintf(out, ",");
    }
    fprintf(out, "\n");
    fprintf(out, "filter_height=");
    for(int i=0;i<M*N;i++) {
        fprintf(out, "%d",filter_height[i]);
        if (i != M*N-1) fprintf(out, ",");
    }
    fprintf(out, "\n");
    
    // bool* enable_maxpooling,int* pooling_stride_x,int* pooling_stride_y,int* pooling_width,int* pooling_height,
    // int* padding_width, int* padding_height,
    fprintf(out, "enable_maxpooling=");
    for(int i=0;i<M;i++) {
        fprintf(out, "%d",enable_maxpooling[i]);
        if (i != M-1) fprintf(out, ",");
    }
    fprintf(out, "\n");
    fprintf(out, "pooling_stride_x=");
    for(int i=0;i<M;i++) {
        fprintf(out, "%d",pooling_stride_x[i]);
        if (i != M-1) fprintf(out, ",");
    }
    fprintf(out, "\n");
    fprintf(out, "pooling_stride_y=");
    for(int i=0;i<M;i++) {
        fprintf(out, "%d",pooling_stride_y[i]);
        if (i != M-1) fprintf(out, ",");
    }
    fprintf(out, "\n");
    fprintf(out, "pooling_width=");
    for(int i=0;i<M;i++) {
        fprintf(out, "%d",pooling_width[i]);
        if (i != M-1) fprintf(out, ",");
    }
    fprintf(out, "\n");
    fprintf(out, "pooling_height=");
    for(int i=0;i<M;i++) {
        fprintf(out, "%d",pooling_height[i]);
        if (i != M-1) fprintf(out, ",");
    }
    fprintf(out, "\n");

    fprintf(out, "padding_width=");
    for(int i=0;i<M;i++) {
        fprintf(out, "%d",padding_width[i]);
        if (i != M-1) fprintf(out, ",");
    }
    fprintf(out, "\n");
    fprintf(out, "padding_height=");
    for(int i=0;i<M;i++) {
        fprintf(out, "%d",padding_height[i]);
        if (i != M-1) fprintf(out, ",");
    }
    fprintf(out, "\n");

    for(int i=0;i<M;i++) {
        for(int j=0;j<N;j++) {
            for(int k=0;k<filter_number[i*N+j];k++) {
                fprintf(out,"F[%d][%d][%d] %d %d %d\n",i,j,k,
                    F[i][j][k]->depth,F[i][j][k]->height,F[i][j][k]->width);
                write3DMatrix(out,F[i][j][k]);
            }
        }
    }

    for(int i=0;i<M;i++) {
        for(int j=0;j<N;j++) {
            for(int k=0;k<filter_number[i*N+j];k++) {
                fprintf(out,"b[%d][%d][%d] %d %d %d\n",i,j,k,
                    b[i][j][k]->depth,F[i][j][k]->height,F[i][j][k]->width);
                write3DMatrix(out,b[i][j][k]);
            }
        }
    }

    for(int i=0;i<K;i++) {
        fprintf(out,"Ws[%d] %d %d\n",i,
            Ws[i]->height,Ws[i]->width);
        write2DMatrix(out,Ws[i]);
    }

    for(int i=0;i<K;i++) {
        fprintf(out,"bs[%d] %d %d\n",i,
            bs[i]->height,bs[i]->width);
        write2DMatrix(out,bs[i]);
    }
    
    fclose(out);
    printf("INFO: Network parameters dumped to %s\n",out_file);
    
    return 0;
}

int loadConvnetConfig(int* M,int* N,
    int** filter_number,int** filter_stride_x, int** filter_stride_y, int** filter_width, int** filter_height, 
    bool** enable_maxpooling,int** pooling_stride_x,int** pooling_stride_y,int** pooling_width,int** pooling_height,
    int** padding_width, int** padding_height,
    float* alpha, bool* normalize_data_per_channel, int* K,
    ThreeDMatrix***** F,ThreeDMatrix***** b,
    TwoDMatrix*** Ws,TwoDMatrix*** bs,
    char* dir) {
    int file_name_length = strlen(dir) + strlen("/convnet.params") + 10;
    char* filename = malloc(sizeof(char)*file_name_length);
    strcpy(filename,dir);
    strcat(filename,"/convnet.params");
    printf("INFO: Loading network parameters from %s\n",filename);
    FILE* fp = fopen(filename,"r");
    if (fp == NULL) {
        printf("ERROR: Cannot open %s to read\n",filename);
        exit(1);
    }
    char** key_values = malloc(sizeof(char*)*2);
    key_values[0] = (char*) malloc(sizeof(char)*100);
    key_values[1] = (char*) malloc(sizeof(char)*100);
    for(int i=0;i<5;i++) {
        getKeyValueFromFile(fp,key_values);
        if (! strcmp(key_values[0],"M")) {
            *M = strtol(key_values[1],NULL,10);
        } else if (! strcmp(key_values[0],"N")) {
            *N = strtof(key_values[1],NULL);
        } else if (! strcmp(key_values[0],"K")) {
            *K = strtof(key_values[1],NULL);
        } else if (! strcmp(key_values[0],"alpha")) {
            *alpha = strtof(key_values[1],NULL);
        } else if (! strcmp(key_values[0],"normalize_data_per_channel")) {
            *normalize_data_per_channel = strtof(key_values[1],NULL);
        } else {
            printf("ERROR: Unrecognized keyword: %s, ignored\n",key_values[0]);
        }
    }
    *filter_number = (int*) malloc(sizeof(int)*(*M)*(*N));
    *filter_stride_x = (int*) malloc(sizeof(int)*(*M)*(*N));
    *filter_stride_y = (int*) malloc(sizeof(int)*(*M)*(*N));
    *filter_width = (int*) malloc(sizeof(int)*(*M)*(*N));
    *filter_height = (int*) malloc(sizeof(int)*(*M)*(*N));
    *enable_maxpooling = (bool*) malloc(sizeof(bool)*(*M));
    int* enable_maxpooling_tmp = malloc(sizeof(int)*(*M));
    *pooling_stride_x = (int*) malloc(sizeof(int)*(*M));
    *pooling_stride_y = (int*) malloc(sizeof(int)*(*M));
    *pooling_width = (int*) malloc(sizeof(int)*(*M));
    *pooling_height = (int*) malloc(sizeof(int)*(*M));
    *padding_width = (int*) malloc(sizeof(int)*(*M));
    *padding_height = (int*) malloc(sizeof(int)*(*M));
    for(int i=0;i<12;i++) {
        if (! strcmp(key_values[0],"filter_number")) {
            splitStr(key_values[1],*filter_number);
        } else if (! strcmp(key_values[0],"filter_stride_x")) {
            splitStr(key_values[1],*filter_stride_x);
        } else if (! strcmp(key_values[0],"filter_stride_y")) {
            splitStr(key_values[1],*filter_stride_y);
        } else if (! strcmp(key_values[0],"filter_width")) {
            splitStr(key_values[1],*filter_width);
        } else if (! strcmp(key_values[0],"filter_height")) {
            splitStr(key_values[1],*filter_height);
        } else if (! strcmp(key_values[0],"enable_maxpooling")) {
            splitStr(key_values[1],enable_maxpooling_tmp);
            for(int i=0;i<*M;i++) {
                (*enable_maxpooling)[i] = (bool) enable_maxpooling_tmp[i];
            }
        } else if (! strcmp(key_values[0],"pooling_stride_x")) {
            splitStr(key_values[1],*pooling_stride_x);
        } else if (! strcmp(key_values[0],"pooling_stride_y")) {
            splitStr(key_values[1],*pooling_stride_y);
        } else if (! strcmp(key_values[0],"pooling_width")) {
            splitStr(key_values[1],*pooling_width);
        } else if (! strcmp(key_values[0],"pooling_height")) {
            splitStr(key_values[1],*pooling_height);
        } else if (! strcmp(key_values[0],"padding_width")) {
            splitStr(key_values[1],*padding_width);
        } else if (! strcmp(key_values[0],"padding_height")) {
            splitStr(key_values[1],*padding_height);
        } else {
            printf("ERROR: Unrecognized keyword: %s, ignored\n",key_values[0]);
        }
    }
    
    *F = (ThreeDMatrix****) malloc(sizeof(ThreeDMatrix***)*(*M));
    *b = (ThreeDMatrix****) malloc(sizeof(ThreeDMatrix***)*(*M));
    for(int i=0;i<*M;i++) {
        (*F)[i] = (ThreeDMatrix***) malloc(sizeof(ThreeDMatrix**)*(*N));
        (*b)[i] = (ThreeDMatrix***) malloc(sizeof(ThreeDMatrix**)*(*N));
        for(int j=0;j<*N;j++) {
            (*F)[i][j] = (ThreeDMatrix**) malloc(sizeof(ThreeDMatrix*)*(*filter_number)[i*(*N)+j]);
            (*b)[i][j] = (ThreeDMatrix**) malloc(sizeof(ThreeDMatrix*)*((*filter_number)[i*(*N)+j]));
        }
    }

    *Ws = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*(*K));
    *bs = (TwoDMatrix**) malloc(sizeof(TwoDMatrix*)*(*K));
    for(int i=0;i<*K;i++) {
        (*Ws)[i] = matrixMalloc(sizeof(TwoDMatrix));
        (*bs)[i] = matrixMalloc(sizeof(TwoDMatrix));
    }
    

    for(int i=0;i<*M;i++) {
        for(int j=0;j<*N;j++) {
            for(int k=0;k<(*filter_number)[i*(*N)+j];k++) {
                (*F)[i][j][k] = load3DMatrix(fp);
            }
        }
    }

    for(int i=0;i<*M;i++) {
        for(int j=0;j<*N;j++) {
            for(int k=0;k<(*filter_number)[i*(*N)+j];k++) {
                (*b)[i][j][k] = load3DMatrix(fp);
            }
        }
    }

    for(int i=0;i<*K;i++) {
        (*Ws)[i] = load2DMatrix(fp);
    }

    for(int i=0;i<*K;i++) {
        (*bs)[i] = load2DMatrix(fp);
    }

    fclose(fp);
    free(key_values[0]);
    free(key_values[1]);
    free(key_values);
    return 0;
}

int CSS2Array(char* str, int* array) {
    int data[8192];
    int counter = 0;
    char* n = malloc(sizeof(char)*8192);
    strcpy(n,str);
    char* sizes_ptr = n;
    for(char* token = strsep(&sizes_ptr, ","); token != NULL; token = strsep(&sizes_ptr, ",")) {
        if (token[0] != '\0') {
            data[counter] = strtol(token,NULL,10);
            counter++;
        }
    }
    
    array = (int*) malloc(sizeof(int)*counter);
    for(int i=0;i<counter;i++) {
        
        array[i] = data[i];
    }
    free(n);
    return counter;
}

ConvnetParameters* readConvnetConfigFile(char* filename) {
    FILE* fp = fopen(filename,"r");
    if (fp == NULL) {
        printf("ERROR: Cannot open %s to read\n",filename);
        exit(1);
    }
    ConvnetParameters* convnet_params = malloc(sizeof(ConvnetParameters));
    memset(convnet_params, 0, sizeof(ConvnetParameters));
    convnet_params->fcnet_param = (FCParameters*) malloc(sizeof(FCParameters));
    // Assign default values 
    convnet_params->shuffle_training_samples = true;
    convnet_params->vertically_flip_training_samples = true;
    convnet_params->horizontally_flip_training_samples = true;
    convnet_params->alpha = 1e-3;
    convnet_params->learning_rate = 0.001;
    convnet_params->verbose = false;
    convnet_params->use_rmsprop = true;
    convnet_params->rmsprop_decay_rate = 0.9;
    convnet_params->rmsprop_eps = 1e-5;
    convnet_params->fcnet_param->use_rmsprop = true;
    convnet_params->fcnet_param->decay_rate = 0.9;
    convnet_params->fcnet_param->eps = 1e-5;
    convnet_params->normalize_data_per_channel = true;
    convnet_params->enable_learning_rate_step_decay = false;
    convnet_params->enable_learning_rate_exponential_decay = false;
    convnet_params->enable_learning_rate_invert_t_decay = false;
    convnet_params->learning_rate_decay_unit = 250;
    convnet_params->learning_rate_decay_a0 = 1.0;
    convnet_params->learning_rate_decay_k = 0.9;
    convnet_params->fcnet_param->learning_rate = convnet_params->learning_rate;

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
    bool number_of_samples_defined = false;
    bool M_defined = false;
    bool N_defined = false;
    bool K_defined = false;
    bool filter_stride_y_defined = false;
    bool filter_stride_x_defined = false;
    bool filter_height_defined = false;
    bool filter_width_defined = false;
    bool filter_number_defined = false;
    bool enable_maxpooling_defined = false;
    bool pooling_height_defined = false;
    bool pooling_width_defined = false;
    bool pooling_stride_y_defined = false;
    bool pooling_stride_x_defined = false;
    bool padding_height_defined = false;
    bool padding_width_defined = false;
    while (! feof(fp)) {
        key_values[0][0] = '\0';
        key_values[1][0] = '\0';
        getKeyValueFromFile(fp,key_values);
        if (key_values[0][0] == '#' || key_values[0][0] == '\0') {
            continue;
        }
        if (! strcmp(key_values[0],"data_set")) {
            if (! number_of_samples_defined) {
                printf("ERROR: number_of_samples must be defined before data_set\n");
            }
            convnet_params->X = load3DMatricesFromFile(key_values[1],convnet_params->number_of_samples);
            data_set_defined = true;
        } else if (! strcmp(key_values[0],"correct_labels")) {
            convnet_params->fcnet_param->correct_labels = load2DMatrixFromFile(key_values[1]);
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
            convnet_params->fcnet_param->hidden_layer_sizes = (int*) malloc(sizeof(int)*network_depth);
            for(int i=0;i<network_depth;i++) {
                convnet_params->fcnet_param->hidden_layer_sizes[i] = layers[i];
            }
            convnet_params->fcnet_param->network_depth = network_depth;
            hidden_layer_sizes_defined = true;
            K_defined = true;
            free(sizes);
        /*
        } else if (! strcmp(key_values[0],"filter_number")) { 
            int data[8192];
            int counter = 0;
            char* n = malloc(sizeof(char)*8192);
            strcpy(n,key_values[1]);
            char* sizes_ptr = n;
            for(char* token = strsep(&sizes_ptr, ","); token != NULL; token = strsep(&sizes_ptr, ",")) {
                if (token[0] != '\0') {
                    data[counter] = strtol(token,NULL,10);
                    counter++;
                }
            }
            
            convnet_params->filter_number = (int*) malloc(sizeof(int)*counter);
            for(int i=0;i<counter;i++) {
                
                convnet_params->filter_number[i] = data[i];
            }
            filter_number_defined = true;
            free(n);
        } else if (! strcmp(key_values[0],"filter_stride_x")) { 
            int data[8192];
            int counter = 0;
            char* n = malloc(sizeof(char)*8192);
            strcpy(n,key_values[1]);
            char* sizes_ptr = n;
            for(char* token = strsep(&sizes_ptr, ","); token != NULL; token = strsep(&sizes_ptr, ",")) {
                if (token[0] != '\0') {
                    data[counter] = strtol(token,NULL,10);
                    counter++;
                }
            }
            
            convnet_params->filter_stride_x = (int*) malloc(sizeof(int)*counter);
            for(int i=0;i<counter;i++) {
                
                convnet_params->filter_stride_x[i] = data[i];
            }
            filter_stride_x_defined = true;
            free(n);
        } else if (! strcmp(key_values[0],"filter_stride_y")) { 
            int data[8192];
            int counter = 0;
            char* n = malloc(sizeof(char)*8192);
            strcpy(n,key_values[1]);
            char* sizes_ptr = n;
            for(char* token = strsep(&sizes_ptr, ","); token != NULL; token = strsep(&sizes_ptr, ",")) {
                if (token[0] != '\0') {
                    data[counter] = strtol(token,NULL,10);
                    counter++;
                }
            }
            
            convnet_params->filter_stride_y = (int*) malloc(sizeof(int)*counter);
            for(int i=0;i<counter;i++) {
                
                convnet_params->filter_stride_y[i] = data[i];
            }
            filter_stride_y_defined = true;
            free(n);
        } else if (! strcmp(key_values[0],"filter_stride")) { 
            int data[8192];
            int counter = 0;
            char* n = malloc(sizeof(char)*8192);
            strcpy(n,key_values[1]);
            char* sizes_ptr = n;
            for(char* token = strsep(&sizes_ptr, ","); token != NULL; token = strsep(&sizes_ptr, ",")) {
                if (token[0] != '\0') {
                    data[counter] = strtol(token,NULL,10);
                    counter++;
                }
            }
            
            convnet_params->filter_stride_x = (int*) malloc(sizeof(int)*counter);
            convnet_params->filter_stride_y = (int*) malloc(sizeof(int)*counter);
            for(int i=0;i<counter;i++) {
                
                convnet_params->filter_stride_x[i] = data[i];
                convnet_params->filter_stride_y[i] = data[i];
            }
            filter_stride_x_defined = true;
            filter_stride_y_defined = true;
            free(n);
        */
        } else if (! strcmp(key_values[0],"number_of_samples")) { 
            convnet_params->number_of_samples = strtol(key_values[1],NULL,10);
            number_of_samples_defined = true;
        } else if (! strcmp(key_values[0],"filter_number")) { 
            CSS2Array(key_values[1],convnet_params->filter_number);
            filter_number_defined = true;
        } else if (! strcmp(key_values[0],"filter_stride_x")) { 
            CSS2Array(key_values[1],convnet_params->filter_stride_x);
            filter_stride_x_defined = true;
        } else if (! strcmp(key_values[0],"filter_stride_y")) { 
            CSS2Array(key_values[1],convnet_params->filter_stride_y);
            filter_stride_y_defined = true;
        } else if (! strcmp(key_values[0],"filter_stride")) { 
            CSS2Array(key_values[1],convnet_params->filter_stride_x);
            filter_stride_x_defined = true;
            CSS2Array(key_values[1],convnet_params->filter_stride_y);
            filter_stride_y_defined = true;
        } else if (! strcmp(key_values[0],"filter_height")) { 
            CSS2Array(key_values[1],convnet_params->filter_height);
            filter_height_defined = true;
        } else if (! strcmp(key_values[0],"filter_width")) { 
            CSS2Array(key_values[1],convnet_params->filter_width);
            filter_width_defined = true;
        } else if (! strcmp(key_values[0],"filter_size")) { 
            CSS2Array(key_values[1],convnet_params->filter_height);
            filter_height_defined = true;
            CSS2Array(key_values[1],convnet_params->filter_width);
            filter_width_defined = true;
        } else if (! strcmp(key_values[0],"enable_maxpooling")) { 
            int* tmp = NULL;
            CSS2Array(key_values[1],tmp);
            convnet_params->enable_maxpooling = malloc(sizeof(bool)*convnet_params->M);
            for(int i=0;i<convnet_params->M;i++) {
                convnet_params->enable_maxpooling[i] = (bool) tmp[i];
            }
            free(tmp);
            enable_maxpooling_defined = true;
        } else if (! strcmp(key_values[0],"pooling_stride_x")) { 
            CSS2Array(key_values[1],convnet_params->pooling_stride_x);
            pooling_stride_x_defined = true;
        } else if (! strcmp(key_values[0],"pooling_stride_y")) { 
            CSS2Array(key_values[1],convnet_params->pooling_stride_y);
            pooling_stride_y_defined = true;
        } else if (! strcmp(key_values[0],"pooling_stride")) { 
            CSS2Array(key_values[1],convnet_params->pooling_stride_x);
            pooling_stride_x_defined = true;
            CSS2Array(key_values[1],convnet_params->pooling_stride_y);
            pooling_stride_y_defined = true;
        } else if (! strcmp(key_values[0],"pooling_width")) { 
            CSS2Array(key_values[1],convnet_params->pooling_width);
            pooling_width_defined = true;
        } else if (! strcmp(key_values[0],"pooling_height")) { 
            CSS2Array(key_values[1],convnet_params->pooling_height);
            pooling_height_defined = true;
        } else if (! strcmp(key_values[0],"pooling_size")) { 
            CSS2Array(key_values[1],convnet_params->pooling_width);
            pooling_width_defined = true;
            CSS2Array(key_values[1],convnet_params->pooling_height);
            pooling_height_defined = true;
        } else if (! strcmp(key_values[0],"padding_width")) { 
            CSS2Array(key_values[1],convnet_params->padding_width);
            padding_width_defined = true;
        } else if (! strcmp(key_values[0],"padding_height")) { 
            CSS2Array(key_values[1],convnet_params->padding_height);
            padding_height_defined = true;
        } else if (! strcmp(key_values[0],"padding_size")) { 
            CSS2Array(key_values[1],convnet_params->padding_width);
            padding_width_defined = true;
            CSS2Array(key_values[1],convnet_params->padding_height);
            padding_height_defined = true;
        } else if (! strcmp(key_values[0],"M")) {
            convnet_params->M = strtol(key_values[1],NULL,10);
            M_defined = true;
        } else if (! strcmp(key_values[0],"N")) {
            convnet_params->N = strtol(key_values[1],NULL,10);
            N_defined = true;
        } else if (! strcmp(key_values[0],"labels")) {
            convnet_params->fcnet_param->labels = strtol(key_values[1],NULL,10);
            labels_defined = true;
        } else if (! strcmp(key_values[0],"minibatch_size")) {
            convnet_params->minibatch_size = strtol(key_values[1],NULL,10);
            minibatch_defined = true;
        } else if (! strcmp(key_values[0],"alpha")) {
            convnet_params->alpha = strtof(key_values[1],NULL);
        } else if (! strcmp(key_values[0],"learing_rate")) {
            convnet_params->learning_rate = strtof(key_values[1],NULL);
        } else if (! strcmp(key_values[0],"epochs")) {
            convnet_params->epochs = strtol(key_values[1],NULL,10);
            epochs_defined = true;
        } else if (! strcmp(key_values[0],"vertically_flip_training_samples")) {
            convnet_params->vertically_flip_training_samples = strtol(key_values[1],NULL,10);
        } else if (! strcmp(key_values[0],"horizontally_flip_training_samples")) {
            convnet_params->horizontally_flip_training_samples = strtol(key_values[1],NULL,10);
        } else if (! strcmp(key_values[0],"rmsprop_decay_rate")) {
            convnet_params->rmsprop_decay_rate = strtof(key_values[1],NULL);
        } else if (! strcmp(key_values[0],"rmsprop_eps")) {
            convnet_params->rmsprop_eps = strtof(key_values[1],NULL);
        } else if (! strcmp(key_values[0],"normalize_data_per_channel")) {
            convnet_params->normalize_data_per_channel= strtol(key_values[1],NULL,10);
        } else if (! strcmp(key_values[0],"enable_learning_rate_step_decay")) {
            convnet_params->enable_learning_rate_step_decay = strtol(key_values[1],NULL,10);
        } else if (! strcmp(key_values[0],"learning_rate_decay_unit")) {
            convnet_params->learning_rate_decay_unit = strtol(key_values[1],NULL,10);
        } else if (! strcmp(key_values[0],"learning_rate_decay_a0")) {
            convnet_params->learning_rate_decay_a0 = strtof(key_values[1],NULL);
        } else if (! strcmp(key_values[0],"learning_rate_decay_k")) {
            convnet_params->learning_rate_decay_k = strtof(key_values[1],NULL);
        } else if (! strcmp(key_values[0],"verbose")) {
            convnet_params->verbose = strtol(key_values[1],NULL,10);
        } else if (! strcmp(key_values[0],"use_rmsprop")) {
            convnet_params->use_rmsprop = strtol(key_values[1],NULL,10);
        } else if (! strcmp(key_values[0],"shuffle_training_samples")) {
            convnet_params->shuffle_training_samples = strtof(key_values[1],NULL);
        } else if (! strcmp(key_values[0],"mode")) {
            convnet_params->mode = (char*) malloc(sizeof(char)*strlen(key_values[1]));
            strcpy(convnet_params->mode,key_values[1]);
            mode_defined = true;
        } else if (! strcmp(key_values[0],"params_dir")) {
            convnet_params->params_save_dir = (char*) malloc(sizeof(char)*strlen(key_values[1]));
            strcpy(convnet_params->params_save_dir,key_values[1]);
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
    if (! number_of_samples_defined) {
        printf("ERROR: number_of_samples not defined\n");
    }
    if (! params_dir_defined) {
        printf("ERROR: Dir to load or save params not defined\n");
        exit(1);
    }
    if (! data_set_defined) {
        printf("ERROR: Data set not defined\n");
        exit(1);
    }
    if (! correct_labels_defined && strcmp(convnet_params->mode,"test")) {
        printf("ERROR: Correct labels not defined\n");
        exit(1);
    }
    if (! hidden_layer_sizes_defined && strcmp(convnet_params->mode,"test")) {
        printf("ERROR: Sizes of hidden layers not defined\n");
        exit(1);
    }
    if (! minibatch_defined) {
        printf("ERROR: Minibatch size not defined\n");
        exit(1);
    }
    if (! labels_defined && strcmp(convnet_params->mode,"test")) {
        printf("ERROR: Number of lables not defined\n");
        exit(1);
    }
    if (! epochs_defined && strcmp(convnet_params->mode,"test")) {
        printf("ERROR: Epochs not defined\n");
        exit(1);
    }
    if (! M_defined) {
        printf("ERROR: M is not defined\n");
        exit(1);
    }
    if (! N_defined) {
        printf("ERROR: N is not defined\n");
        exit(1);
    }
    if (! K_defined) {
        printf("ERROR: Sizes of hidden layers in fully connected network are not defined\n");
        exit(1);
    }
    if (! filter_stride_y_defined) {
        printf("ERROR: Strides of filters in Y direction are not defined\n");
        exit(1);
    }
    if (! filter_stride_x_defined) {
        printf("ERROR: Strides of filters in X direction are not defined\n");
        exit(1);
    }
    if (! filter_height_defined) {
        printf("ERROR: Heights of filters are not defined\n");
        exit(1);
    }
    if (! filter_width_defined) {
        printf("ERROR: Widths of filters are not defined\n");
        exit(1);
    }
    if (! filter_number_defined) {
        printf("ERROR: Numbers of filters in each convolutional layer are not defined\n");
        exit(1);
    }
    if (! enable_maxpooling_defined) {
        printf("ERROR: No info provided to enable max pooling in each convolutional layer\n");
        exit(1);
    }
    if (! pooling_height_defined) {
        printf("ERROR: Heights of pooling windows are not defined\n");
        exit(1);
    }
    if (! pooling_width_defined) {
        printf("ERROR: Widths of pooling windows not defined\n");
        exit(1);
    }
    if (! pooling_stride_y_defined) {
        printf("ERROR: Strides of pooling windows in Y direction are not defined\n");
        exit(1);
    }
    if (! pooling_stride_x_defined) {
        printf("ERROR: Strides of pooling windows in Y direction are not defined\n");
        exit(1);
    }
    if (! padding_height_defined) {
        printf("ERROR: Heights of zero padding in each layers are not defined\n");
        exit(1);
    }
    if (! padding_width_defined) {
        printf("ERROR: Widths of zero padding in each layers are not defined\n");
        exit(1);
    }
    convnet_params->fcnet_param->hidden_layer_sizes[convnet_params->fcnet_param->network_depth-1] = convnet_params->fcnet_param->labels;
    convnet_params->fcnet_param->use_rmsprop = convnet_params->use_rmsprop;
    convnet_params->fcnet_param->decay_rate = convnet_params->rmsprop_decay_rate;
    convnet_params->fcnet_param->eps = convnet_params->rmsprop_eps;
    convnet_params->fcnet_param->learning_rate = convnet_params->learning_rate;
    convnet_params->fcnet_param->enable_learning_rate_step_decay = convnet_params->enable_learning_rate_step_decay;
    convnet_params->fcnet_param->enable_learning_rate_exponential_decay = convnet_params->enable_learning_rate_exponential_decay;
    convnet_params->fcnet_param->enable_learning_rate_invert_t_decay = convnet_params->enable_learning_rate_invert_t_decay;
    convnet_params->fcnet_param->learning_rate_decay_unit = convnet_params->learning_rate_decay_unit;
    convnet_params->fcnet_param->learning_rate_decay_a0 = convnet_params->learning_rate_decay_a0;
    convnet_params->fcnet_param->learning_rate_decay_k = convnet_params->learning_rate_decay_k;
    return convnet_params;
}
