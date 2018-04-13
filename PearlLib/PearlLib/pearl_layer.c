#include <pearl_layer.h>

void pearl_layer_initialise(pearl_layer *layer, const pearl_layer *prev_layer)
{
    if (layer) {
        if (prev_layer) {
            if (layer->biases == NULL) {
                layer->biases = pearl_vector_create(layer->neurons);
            }
            if (layer->weights == NULL) {
                layer->weights = pearl_matrix_create(layer->neurons, prev_layer->neurons);
                double scale = 1.0;
                //https://arxiv.org/abs/1704.08863
                switch (layer->activation_function) {
                    case pearl_activation_function_type_linear:
                        scale = 1.0 / prev_layer->neurons;
                        break;
                    case pearl_activation_function_type_sigmoid:
                        scale = 3.6 / sqrt(prev_layer->neurons);
                        break;
                    case pearl_activation_function_type_tanh:
                        //scale = 1.0 / sqrt(prev_layer->neurons);
                        //http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf?hc_location=ufi
                        scale = sqrt(6.0 / (layer->neurons + prev_layer->neurons));
                        break;
                }
                for (int i = 0; i < layer->weights->m * layer->weights->n; i++) {
                    layer->weights->data[i] = -1.0 + ((float)rand() / (float)(RAND_MAX)) * scale * 2.0;
                }
                pearl_layer_print(layer);
            }
        }
    }
}

void pearl_layer_destroy(pearl_layer *layer)
{
    if (layer) {
        if (layer) {
            pearl_vector_destroy(layer->biases);
        }
        if (layer->weights) {
            pearl_matrix_destroy(layer->weights);
        }
    }
}

void pearl_layer_print(pearl_layer *layer)
{
    if (layer) {
        printf("Type: pearl_layer\n");
        printf("Type: ");
        switch (layer->type) {
            case pearl_layer_type_input:
                printf("Input");
                break;
            case pearl_layer_type_fully_connect:
                printf("Fully connect");
                break;
            case pearl_layer_type_output:
                printf("Output");
                break;
            case pearl_layer_type_dropout:
                printf("Dropout");
                break;
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
            pearl_matrix_print(layer->weights);
        }
        else {
            printf("None\n");
        }

        printf("Biases: ");
        if (layer->biases) {
            for (int i = 0; i < layer->biases->n; i++) {
                printf("%f ", layer->biases->data[i]);
            }
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

void pearl_layer_forward(pearl_layer *layer, const pearl_matrix *input, pearl_matrix *z, pearl_matrix *a)
{
    assert(input->n == layer->weights->n);
    double (*activationFunctionPtr)(double) = pearl_activation_function_pointer(layer->activation_function);

    for (int i = 0; i < input->m; i++) {
        for (int j = 0; j < layer->weights->n; j++) {
            double sum = 0;
            for (int k = 0; k < layer->weights->m; k++) {
                sum += input->data[ARRAY_IDX(i, k, input->n)] * layer->weights->data[ARRAY_IDX(k, j, layer->weights->n)];
            }
            sum += layer->biases->data[j];
            z->data[ARRAY_IDX(i, j, z->n)] = sum;
            a->data[ARRAY_IDX(i, j, a->n)] = (*activationFunctionPtr)(sum);
        }
    }
}
