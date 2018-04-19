#include <pearl_tensor.h>

PEARL_API pearl_tensor *pearl_tensor_create(int num_args, ...)
{
    va_list list;
    pearl_tensor *result = malloc(sizeof(pearl_tensor));
    result->dimension = num_args;
    result->size = calloc(num_args, sizeof(int));
    va_start(list, num_args);
    int alloc = 1;
    for( int i = 0 ; i < num_args; i++ )
    {
        int arg = va_arg( list, int );
        result->size[i] = arg;
        alloc *= arg;
    }
    va_end(list);
    result->data = calloc(alloc, sizeof(double));
    return result;
}

PEARL_API void pearl_tensor_destroy(pearl_tensor *x)
{
    if (x) {
        if (x->size) {
            free(x->size);
        }
        if (x->data) {
            free(x->data);
        }
        free(x);
    }
}

PEARL_API pearl_tensor *pearl_tensor_copy(const pearl_tensor *x){
    pearl_tensor *result = malloc(sizeof(pearl_tensor));
    result->dimension = x->dimension;
    result->size = calloc(x->dimension, sizeof(int));
    int alloc = 1;
    for( int i = 0 ; i < x->dimension; i++ )
    {
        result->size[i] = x->size[i];
        alloc *= x->size[i];
    }
    result->data = calloc(alloc, sizeof(double));

    int lenght = 1;
    for(int i=0; i<x->dimension; i++){
        lenght += x->size[i];
    }

    for(int i=0; i<lenght; i++){
        result->data[i] = x->data[i];
    }
    return result;
}

PEARL_API void pearl_tensor_print(pearl_tensor *x){
    switch (x->dimension) {
    case 1:
        printf("Vector of %d units\n", x->size[0]);
        for(int i=0; i<x->size[0];i++){
            printf("%f\n", x->data[i]);
        }
        break;
    case 2:
        printf("Matrix of %d x %d units\n", x->size[0], x->size[1]);
        for(int i=0; i<x->size[0];i++){
            for(int j=0; j<x->size[1];j++){
                printf("%f ", x->data[ARRAY_IDX_2D(i,j, x->size[1])]);
            }
            printf("\n");
        }
        break;
    default:
        printf("Cannot print tensor of dimension %d\n", x->dimension);
        break;
    }
}
