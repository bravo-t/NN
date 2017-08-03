#include <malloc.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "matrix_type.h"
#include "misc_utils.h"

TwoDMatrix* matrixMalloc(int size) {
    TwoDMatrix* M = malloc(size);
    memset(M,0,size);
    M->initialized = false;
    return M;
}

int dumpLearnableParams(TwoDMatrix** Ws, TwoDMatrix** bs) {
    return 0;
}

int loadLearnableParams(TwoDMatrix** Ws, TwoDMatrix** bs) {
    return 0;
}


