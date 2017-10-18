#ifndef __MISC_UTILS_HEADER___
#define __MISC_UTILS_HEADER___

#ifdef DEBUG
#define debugPrintMatrix(M) __debugPrintMatrix(M,#M)
#else
#define debugPrintMatrix(M) ((void)0)
#endif

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
char determineMemoryUnit(unsigned int n);
float memoryUsageReadable(unsigned long long int n, char unit);
ThreeDMatrix* load3DMatrixFromFile(char* filename);
float matrixError3D(ThreeDMatrix* a, ThreeDMatrix* b);
void print3DMatrix(ThreeDMatrix *M);
void check3DMatrixDiff(ThreeDMatrix* a, ThreeDMatrix* b, float thres);
int writeImage(ThreeDMatrix* X, char* var_name, char* img_dir);
int verticallyFlipSample(ThreeDMatrix* in, ThreeDMatrix* out);
int horizontallyFlipSample(ThreeDMatrix* in, ThreeDMatrix* out);
int shuffleTrainingSamples(ThreeDMatrix** data_in, 
    TwoDMatrix* label_in,
    int number_of_samples, 
    bool vertically_flip_samples,
    bool horizontally_flip_samples,
    ThreeDMatrix** data_out,
    TwoDMatrix* label_out);

#endif
