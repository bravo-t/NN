#include "matrix_operations.h"
#include <stdlib.h>

int affineForwardLayer(TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* b, TwoDMatrix* OUT) {
    init2DMatrix(OUT, X->height, W->width);
    if (dotProduct(X,W,OUT)) {
        printf("ERROR: Input matrix size mismatch: X->width = %d, W->height = %d\n", X->width,W->height);
        exit 1;
    }

}
