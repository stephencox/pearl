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

json_object *pearl_tensor_to_json(pearl_tensor *tensor)
{
    json_object *json_obj = json_object_new_object();
    json_object_object_add(json_obj, "version", pearl_version_to_json(tensor->version));
#ifdef ENV64BIT
    json_object_object_add(json_obj, "dimension", json_object_new_int64(tensor->dimension));
#else
    json_object_object_add(json_obj, "dimension", json_object_new_int(tensor->dimension));
#endif
    json_object *json_size = json_object_new_array();
    for (unsigned int i = 0 ; i < tensor->dimension; i++) {
#ifdef ENV64BIT
        json_object_array_add(json_size, json_object_new_int64(tensor->size[i]));
#else
        json_object_array_add(json_size, json_object_new_int(tensor->size[i]));
#endif
    }
    json_object_object_add(json_obj, "size", json_size);
    json_object *json_data = json_object_new_array();
    unsigned int num_data = 1;
    for (unsigned int i = 0; i < tensor->dimension; i++) {
        num_data *= tensor->size[i];
    }
    for (unsigned int i = 0; i < num_data; i++) {
        json_object_array_add(json_data, json_object_new_double(tensor->data[i]));
    }
    json_object_object_add(json_obj, "data", json_data);
    return json_obj;
}
