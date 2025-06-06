#include <pearl_tensor.h>
#include <string.h> // Add this include

PEARL_API pearl_tensor *pearl_tensor_create(const int num_args, ...)
{
    va_list list;
    pearl_tensor *result = calloc(1, sizeof(pearl_tensor));
    result->dimension = num_args;
    result->size = calloc(num_args, sizeof(unsigned int));
    va_start(list, num_args);
    unsigned int alloc = 1;
    for (int i = 0 ; i < num_args; i++) {
        unsigned int arg = va_arg(list, unsigned int);
        result->size[i] = arg;
        alloc *= arg;
    }
    va_end(list);
    result->data = calloc(alloc, sizeof(float));
    return result;
}

PEARL_API void pearl_tensor_destroy(pearl_tensor **x)
{
    if (*x != NULL) {
        if ((*x)->size != NULL) {
            free((*x)->size);
            (*x)->size = NULL;
        }
        if ((*x)->data != NULL) {
            free((*x)->data);
            (*x)->data = NULL;
        }
        free(*x);
        *x = NULL;
    }
}

PEARL_API pearl_tensor *pearl_tensor_copy(const pearl_tensor *x)
{
    pearl_tensor *result = calloc(1, sizeof(pearl_tensor));
    result->dimension = x->dimension;
    result->size = calloc(x->dimension, sizeof(unsigned int));
    
    unsigned int total_elements = 1;
    for (unsigned int i = 0 ; i < x->dimension; i++) {
        result->size[i] = x->size[i];
        total_elements *= x->size[i];
    }
    
    result->data = calloc(total_elements, sizeof(float));
    if (x->data != NULL && result->data != NULL && total_elements > 0) { // Add checks for safety
        memcpy(result->data, x->data, total_elements * sizeof(float));
    } else if (total_elements == 0) {
        // If total_elements is 0, data might be NULL, which is fine.
        // If x->data is NULL but total_elements > 0, calloc still provides zeroed memory.
    }


    return result;
}

PEARL_API void pearl_tensor_reduce_dimension(pearl_tensor **x, const unsigned int dimension)
{
    pearl_tensor *x_p = (*x);
    assert(x_p->dimension == 2);
    assert(dimension == 1);
    assert(x_p->size[1] == 1 || x_p->size[0] == 1);
    x_p->dimension = dimension;
    if (x_p->size[0] == 1) {
        x_p->size[0] = x_p->size[1];
    }
    x_p->size = realloc(x_p->size, dimension * sizeof(unsigned int));
}
