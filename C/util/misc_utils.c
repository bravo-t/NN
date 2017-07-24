#include "matrix_type.h"
#include "misc_utils.h"
#include <malloc.h>

TwoDMatrix* matrixMalloc(int size) {
	TwoDMatrix* M = malloc(size);
	M->initialized = false;
	return M;
}