#include <pearl_tensor.h>

PEARL_API pearl_tensor *pearl_tensor_create(const int num_args, ...)
{
    va_list list;
    pearl_tensor *result = malloc(sizeof(pearl_tensor));
    result->version.major = PEARL_TENSOR_VERSION_MAJOR;
    result->version.minor = PEARL_TENSOR_VERSION_MINOR;
    result->version.revision = PEARL_TENSOR_VERSION_REVISION;
    result->dimension = num_args;
    result->size = calloc(num_args, sizeof(unsigned int));
    va_start(list, num_args);
    int alloc = 1;
    for (int i = 0 ; i < num_args; i++) {
        int arg = va_arg(list, int);
        result->size[i] = arg;
        alloc *= arg;
    }
    va_end(list);
    result->data = calloc(alloc, sizeof(double));
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
    pearl_tensor *result = malloc(sizeof(pearl_tensor));
    result->dimension = x->dimension;
    result->size = calloc(x->dimension, sizeof(unsigned int));
    int alloc = 1;
    for (unsigned int i = 0 ; i < x->dimension; i++) {
        result->size[i] = x->size[i];
        alloc *= x->size[i];
    }
    result->data = calloc(alloc, sizeof(double));

    unsigned int num_data = 1;
    for (unsigned int i = 0; i < x->dimension; i++) {
        num_data *= x->size[i];
    }

    for (unsigned int i = 0; i < num_data; i++) {
        result->data[i] = x->data[i];
    }
    return result;
}
