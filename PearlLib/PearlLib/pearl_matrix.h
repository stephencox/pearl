#ifndef PEARL_MATRIX_H
#define PEARL_MATRIX_H

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <pearl_global.h>
#include <pearl_vector.h>

struct pearl_matrix {
    int m, n;
    double *data;
};

#define ARRAY_IDX(i,j,n) (i*n+j)

PEARL_API struct pearl_matrix *pearl_matrix_create(int m, int n);
PEARL_API void pearl_matrix_destroy(struct pearl_matrix *x);
PEARL_API void pearl_matrix_print(struct pearl_matrix *x);

#endif // PEARL_MATRIX_H
