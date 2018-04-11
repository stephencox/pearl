#include <pearl_vector.h>

PEARL_API pearl_vector *pearl_vector_create(int n)
{
    pearl_vector *result = malloc(sizeof(pearl_vector));
    result->n = n;
    result->data = calloc(n, sizeof(double));
    return result;
}

PEARL_API void pearl_vector_destroy(pearl_vector *x)
{
    if (x) {
        if (x->data) {
            free(x->data);
        }
        free(x);
    }
}
