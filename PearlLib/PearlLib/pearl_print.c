#include <pearl_print.h>

void pearl_layer_print(const pearl_layer *layer)
{
    if (layer) {
        printf("Type: pearl_layer\n");
        printf("Type: ");
        switch (layer->type) {
            case pearl_layer_type_fully_connect:
                printf("Fully connect");
                break;
            //            case pearl_layer_type_dropout:
            //                printf("Dropout");
            //                break;
            default:
                printf("None");
                break;
        }
        printf("\n");

        printf("Activation: ");
        switch (layer->activation_function) {
            case pearl_activation_function_type_linear:
                printf("Linear");
                break;
            case pearl_activation_function_type_sigmoid:
                printf("Sigmoid");
                break;
            case pearl_activation_function_type_tanh:
                printf("Tanh");
                break;
            default:
                printf("None");
                break;
        }
        printf("\n");

        printf("Weights:\n");
        if (layer->weights) {
            pearl_tensor_print(layer->weights);
        }
        else {
            printf("None\n");
        }

        printf("Biases: ");
        if (layer->biases) {
            pearl_tensor_print(layer->biases);
        }
        else {
            printf("None");
        }
        printf("\n");

    }
    else {
        printf("Layer is NULL");
    }
    printf("\n");
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
