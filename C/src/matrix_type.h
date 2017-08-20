#ifndef __TYPE_HEADER__
#define __TYPE_HEADER__

typedef struct {
    int height;
    int width;
    float** d;
    bool initialized;
} TwoDMatrix;

typedef struct {
	int height;
    int width;
    int depth;
    float*** d;
    bool initialized;
} ThreeDMatrix;

#endif
