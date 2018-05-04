#ifndef PEARL_TENSOR_H
#define PEARL_TENSOR_H

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <stdarg.h>
#include <pearl_global.h>

typedef struct {
    unsigned int dimension;
    unsigned int *size;
    double *data;
} pearl_tensor;

#define ARRAY_IDX_2D(i,j,n) (i*n+j)

PEARL_API pearl_tensor *pearl_tensor_create(const int num_args, ...);
PEARL_API void pearl_tensor_destroy(pearl_tensor **x);
PEARL_API pearl_tensor *pearl_tensor_copy(const pearl_tensor *x);
PEARL_API void pearl_tensor_print(const pearl_tensor *x);
PEARL_API void pearl_tensor_save(const pearl_tensor *x, FILE *f);
PEARL_API pearl_tensor *pearl_tensor_load(FILE *f);

#endif //PEARL_TENSOR_H
