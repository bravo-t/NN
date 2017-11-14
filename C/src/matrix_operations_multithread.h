#ifndef __OPERATION_HEADER_MT__
#define __OPERATION_HEADER_MT__

struct TwoMatrixOperationsRowArgs {
    TwoDMatrix* X;
    TwoDMatrix* M;
    TwoDMatrix* OUT;
    float a;
    float mean;
    float std;
    int n;
    int h_start;
    int h_end;
};

struct ThreeMatrixOperationsRowArgs {
    ThreeDMatrix* X;
    ThreeDMatrix* M;
    ThreeDMatrix* OUT;
    float a;
    float mean;
    float std;
    int n;
    int d_start;
    int d_end;
};

int dotProduct_MT(TwoDMatrix* X, TwoDMatrix* M, TwoDMatrix* OUT, int number_of_threads);
void* dotProductRow(void* args);

#endif