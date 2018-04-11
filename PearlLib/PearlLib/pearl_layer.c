#include <pearl_layer.h>

void pearl_layer_initialise(struct pearl_layer *layer, const struct pearl_layer *prev_layer)
{
    if (layer) {
        if (prev_layer) {
            if (!layer->biases) {
                layer->biases = pearl_vector_create(layer->neurons);
            }
            if (!layer->weights) {
                layer->weights = pearl_vector_create(layer->neurons * prev_layer->neurons);
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
                        scale = 1.0 / sqrt(prev_layer->neurons);
                        break;
                }
                for (int i = 0; i < layer->weights->n; i++) {
                    layer->weights->data[i] = rand() / RAND_MAX * scale;
                }
            }
        }
    }
}

void pearl_layer_destroy(struct pearl_layer *layer)
{
    if (layer) {
        if (layer) {
            pearl_vector_destroy(layer->biases);
        }
        if (layer->weights) {
            pearl_vector_destroy(layer->weights);
        }
    }
}

struct pearl_matrix *pearl_layer_forward(struct pearl_layer *layer, const struct pearl_matrix *input){
    assert(input->m == layer->weights->n);
    assert(input->m == layer->biases->n);
    struct pearl_matrix *result = pearl_matrix_create(input->m, layer->weights->n);

    double (*activationFunctionPtr)(double);
    switch (layer->activation_function) {
    case pearl_activation_function_type_tanh:
        activationFunctionPtr = &tanh;
        break;
    case pearl_activation_function_type_sigmoid:
        activationFunctionPtr = &tanh;
        break;
    case pearl_activation_function_type_linear:
        activationFunctionPtr = &tanh;
        break;
    default:
        activationFunctionPtr = &tanh;
        break;
    }

    for (int i = 0; i < input->m; i++) {
        for (int j = 0; j < layer->weights->n; j++) {
            double sum = 0;
            for (int k = 0; k < layer->weights->n; k++) {
                sum += input->data[ARRAY_IDX(i, k, input->n)] * layer->weights->data[j] + layer->biases->data[j];
            }
            result->data[ARRAY_IDX(i, j, result->n)] = (*activationFunctionPtr)(sum);
        }
    }
    return result;
}
