#ifndef __MISC_UTILS_HEADER___
#define __MISC_UTILS_HEADER___

void write2DMatrix(FILE* fp, TwoDMatrix* M);
void getKeyValueFromFile(FILE* fp, char** retval);
int dumpNetworkConfig(int network_depth, float alpha, TwoDMatrix** Ws, TwoDMatrix** bs, bool use_batchnorm, TwoDMatrix** mean_caches, TwoDMatrix** var_caches, TwoDMatrix** gammas, TwoDMatrix** betas, float eps, char* output_dir);
int loadNetworkConfig(char* dir, int* network_depth, float* alpha, TwoDMatrix*** Ws, TwoDMatrix*** bs, bool* use_batchnorm, TwoDMatrix*** mean_caches, TwoDMatrix*** var_caches, TwoDMatrix*** gammas, TwoDMatrix*** betas, float* batchnorm_eps);
float matrixError(TwoDMatrix* a, TwoDMatrix* b); 
void printMatrix(TwoDMatrix *M);
void __debugPrintMatrix(TwoDMatrix *M, char* name);
void checkMatrixDiff(TwoDMatrix* a, TwoDMatrix* b, float thres);
TwoDMatrix* load2DMatrix(FILE* fp);
TwoDMatrix* load2DMatrixFromFile(char* filename);
FCParameters* readNetworkConfigFile(char* filename);

#endif
