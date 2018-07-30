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

    unsigned int lenght = 1;
    for (unsigned int i = 0; i < x->dimension; i++) {
        lenght += x->size[i];
    }

    for (unsigned int i = 0; i < lenght; i++) {
        result->data[i] = x->data[i];
    }
    return result;
}

PEARL_API void pearl_tensor_print(const pearl_tensor *x)
{
    switch (x->dimension) {
        case 1:
            printf("Vector of %d units\n", x->size[0]);
            for (unsigned int i = 0; i < x->size[0]; i++) {
                printf("%f\n", x->data[i]);
            }
            break;
        case 2:
            printf("Matrix of %d x %d units\n", x->size[0], x->size[1]);
            for (unsigned int i = 0; i < x->size[0]; i++) {
                for (unsigned int j = 0; j < x->size[1]; j++) {
                    printf("%f ", x->data[ARRAY_IDX_2D(i, j, x->size[1])]);
                }
                printf("\n");
            }
            break;
        default:
            printf("Cannot print tensor of dimension %d\n", x->dimension);
            break;
    }
}

PEARL_API void pearl_tensor_save(const pearl_tensor *x, FILE *f)
{
    if(f == NULL){
        fprintf(stderr,"Save error, could not save tensor; file not open!\n");
        return;
    }
    if(x == NULL){
        fprintf(stderr,"Save error, could not save tensor; tensor is NULL!\n");
        return;
    }
    fwrite(&x->dimension, sizeof(unsigned int), 1, f);
    fwrite(&x->size, sizeof(unsigned int), x->dimension, f);
    unsigned int count = 1;
    for (unsigned int i = 0; i < x->dimension; i++) {
        count *= x->size[i];
    }
    fwrite(&x->data, sizeof(double), count, f);
}

PEARL_API pearl_tensor *pearl_tensor_load(FILE *f){
    if(f == NULL){
        return NULL;
    }

    pearl_tensor *x = malloc(sizeof(pearl_tensor));

    size_t count = 0;
    count = fread(&x->dimension, sizeof(unsigned int), 1, f);
    if(count!=1){
        fprintf(stderr,"Load failed: Error reading tensor dimension!\n");
        pearl_tensor_destroy(&x);
        return NULL;
    }
    x->size = calloc(x->dimension, sizeof(unsigned int));
    x->data = NULL;

    count = fread(&x->size, sizeof(unsigned int), x->dimension, f);
    if(count!=x->dimension){
        fprintf(stderr,"Load failed: Error reading tensor sizes!\n");
        pearl_tensor_destroy(&x);
        return NULL;
    }

    unsigned int num_data = 1;
    for (unsigned int i = 0; i < x->dimension; i++) {
        num_data *= x->size[i];
    }

    count = fread(&x->data, sizeof(double), num_data, f);
    if(count!=num_data){
        fprintf(stderr,"Load failed: Error reading tensor data!\n");
        pearl_tensor_destroy(&x);
        return NULL;
    }

    return x;
}
