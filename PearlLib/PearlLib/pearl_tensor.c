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

JSON_Value *pearl_tensor_to_json(pearl_tensor *tensor)
{
    JSON_Value *value = json_value_init_object();
    JSON_Object *obj = json_value_get_object(value);
    json_object_set_value(obj, "version", pearl_version_to_json(tensor->version));
    json_object_set_number(obj, "dimension", tensor->dimension);
    JSON_Value *size = json_value_init_array();
    JSON_Array *size_array = json_value_get_array(size);
    unsigned int num_data = 1;
    for (unsigned int i = 0 ; i < tensor->dimension; i++) {
        json_array_append_number(size_array, tensor->size[i]);
        num_data *= tensor->size[i];
    }
    json_object_set_value(obj, "size", size);
    JSON_Value *data = json_value_init_array();
    JSON_Array *data_array = json_value_get_array(data);
    for (unsigned int i = 0; i < num_data; i++) {
        json_array_append_number(data_array, tensor->data[i]);
    }
    json_object_set_value(obj, "data", data);
    return value;
}

pearl_tensor *pearl_tensor_from_json(JSON_Value *json)
{
    JSON_Object *obj = json_value_get_object(json);
    JSON_Object *tensor_version = json_object_get_value(obj, "version");
    if (tensor_version == NULL) {
        return NULL;
    }
    pearl_tensor *tensor = malloc(sizeof(pearl_tensor));
    tensor->version = pearl_version_from_json(tensor_version);
    tensor->dimension = json_object_get_number(obj, "dimension");

    // SIZE
    JSON_Value *tensor_size_array = json_object_get_value(obj, "size");
    if (tensor_size_array == NULL) {
        pearl_tensor_destroy(&tensor);
        return NULL;
    }
    tensor->size = calloc(tensor->dimension, sizeof(unsigned int));
    unsigned int num_data = 1;
    JSON_Array *size_array = json_value_get_array(tensor_size_array);
    if (size_array == NULL) {
        pearl_tensor_destroy(&tensor);
        return NULL;
    }
    for (unsigned int i = 0; i < tensor->dimension; i++) {
        tensor->size[i] = json_array_get_number(size_array, i);
        num_data *= tensor->size[i];
    }
    JSON_Value *tensor_data_array = json_object_get_value(obj, "data");
    if (tensor_data_array == NULL) {
        pearl_tensor_destroy(&tensor);
        return NULL;
    }
    tensor->data = calloc(num_data, sizeof(double));
    JSON_Array *data_array = json_value_get_array(tensor_data_array);
    if (data_array == NULL) {
        pearl_tensor_destroy(&tensor);
        return NULL;
    }
    for (unsigned int i = 0; i < num_data; i++) {
        tensor->data[i] = json_array_get_number(data_array, i);
    }

    return tensor;
}
