#ifndef __OPERATION_HEADER_MT__
#define __OPERATION_HEADER_MT__

struct DotProductRowArgs {
    TwoDMatrix* X;
    TwoDMatrix* W;
    TwoDMatrix* OUT;
    int h_start;
    int h_end;
};

int dotProduct_MT(TwoDMatrix* X, TwoDMatrix* W, TwoDMatrix* OUT, int number_of_threads);
void* dotProductRow(void* args);

#endif