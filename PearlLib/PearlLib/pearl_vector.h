#ifndef PEARL_VECTOR_H
#define PEARL_VECTOR_H

#include <stdlib.h>
#include <pearl_global.h>

typedef struct {
    int n;
    double *data;
} pearl_vector;

PEARL_API pearl_vector *pearl_vector_create(int n);
PEARL_API void pearl_vector_destroy(pearl_vector *x);

#endif // PEARL_VECTOR_H
