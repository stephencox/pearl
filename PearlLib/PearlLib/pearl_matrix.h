#ifndef PEARL_MATRIX_H
#define PEARL_MATRIX_H

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <cblas.h>
#include <pearl_global.h>

typedef struct {
    int m, n;
    double *data;
} pearl_matrix;

#define ARRAY_IDX(i,j,n) (i*n+j)

PEARL_API pearl_matrix *pearl_matrix_create(int m, int n);
PEARL_API void pearl_matrix_destroy(pearl_matrix *x);
PEARL_API void pearl_matrix_print(pearl_matrix *x);
PEARL_API pearl_matrix *pearl_matrix_muliply_transpose_plain(pearl_matrix *x, pearl_matrix *y);
PEARL_API pearl_matrix *pearl_matrix_muliply_transpose_cblas(pearl_matrix *x, pearl_matrix *y);
#endif // PEARL_MATRIX_H
