#include "layer_utils.h"
#include "matrix_operations.h"

int broadcastMatrix(TwoDMatrix* M, int n, int direction, TwoDMatrix* OUT) {
    if (direction == 0) {
        if (M->width != 1) {
            printf("ERROR: Cannot horizontally broadcast matrix with a width that is not 1\n");
            return 1;
        }
        init2DMatrix(OUT, M->height, n);
        for(int i=0;i<M->height;i++) {
            for(int j=0;j<n;j++) {
                OUT->d[i][j] = M->d[i][0];
            }
        }
    } else {
        if (M->height != 1) {
            printf("ERROR: Cannot vertically broadcast matrix with a height that is not 1\n");
            return 1;
        }
        init2DMatrix(OUT, n, M->width);
        for(int i=0;i<n;i++) {
            for(int j=0;j<M->width;j++) {
                OUT->d[i][j] = M->d[0][j];
            }
        }
    }
    return 0;
}

int BroadcastAdd(TwoDMatrix* M, TwoDMatrix* b, TwoDMatrix* OUT) {
    
}

