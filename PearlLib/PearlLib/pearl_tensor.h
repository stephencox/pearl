#ifndef PEARL_TENSOR_H
#define PEARL_TENSOR_H

#include <stdlib.h>
#include <assert.h>
#include <stdarg.h>
#include <pearl_global.h>

#define PEARL_TENSOR_VERSION_MAJOR 1
#define PEARL_TENSOR_VERSION_MINOR 0
#define PEARL_TENSOR_VERSION_REVISION 0

typedef struct {
    unsigned int dimension;
    unsigned int *size;
    float *data;
} pearl_tensor;

#define ARRAY_IDX_2D(i,j,n) (i*n+j)

PEARL_API pearl_tensor *pearl_tensor_create(const int num_args, ...);
PEARL_API void pearl_tensor_destroy(pearl_tensor **x);
PEARL_API pearl_tensor *pearl_tensor_copy(const pearl_tensor *x);
PEARL_API void pearl_tensor_reduce_dimension(pearl_tensor **x, const unsigned int dimension);

#endif //PEARL_TENSOR_H
