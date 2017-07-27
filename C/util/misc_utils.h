#ifndef __MISC_UTILS_HEADER___
#define __MISC_UTILS_HEADER___

TwoDMatrix* matrixMalloc(int size);
TwoDMatrix* load2DMatrixFromFile(char* filename);
float matrixError(TwoDMatrix* a, TwoDMatrix* b); 

#endif
