#include <pearl_layer.h>

void pearl_layer_initialise(pearl_layer *layer, const pearl_layer *prev_layer)
{
    if (layer) {
        if (prev_layer) {
            if (layer->biases == NULL) {
                layer->biases = pearl_tensor_create(1, layer->neurons);
            }
            if (layer->weights == NULL) {
                layer->weights = pearl_tensor_create(2, layer->neurons, prev_layer->neurons);
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
                for (int i = 0; i < layer->weights->size[0] * layer->weights->size[1]; i++) {
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
            pearl_tensor_destroy(layer->biases);
        }
        if (layer->weights) {
            pearl_tensor_destroy(layer->weights);
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
            //pearl_matrix_print(layer->weights);
        }
        else {
            printf("None\n");
        }

        printf("Biases: ");
        if (layer->biases) {
            for (int i = 0; i < layer->biases->size[0]; i++) {
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

void pearl_layer_forward(pearl_layer *layer, const pearl_tensor *input, pearl_tensor *z, pearl_tensor *a)
{
    assert(input->size[1] == layer->weights->size[1]);
    assert(layer->biases->dimension==1);
    assert(layer->weights->size[0] == layer->biases->size[0]);
    double (*activationFunctionPtr)(double) = pearl_activation_function_pointer(layer->activation_function);

    for (int i = 0; i < input->size[0]; i++) {
        for (int j = 0; j < layer->weights->size[0]; j++) {
            double sum = 0;
            for (int k = 0; k < layer->weights->size[1]; k++) {
                assert(ARRAY_IDX_2D(i, k, input->size[1]) < input->size[0]*input->size[1]);
                assert(ARRAY_IDX_2D(k, j, layer->weights->size[0]) < layer->weights->size[0]*layer->weights->size[1]);
                sum += input->data[ARRAY_IDX_2D(i, k, input->size[1])] * layer->weights->data[ARRAY_IDX_2D(k, j, layer->weights->size[0])];
            }
            sum += layer->biases->data[j];
            assert(ARRAY_IDX_2D(i, j, z->size[1]) < z->size[0]*z->size[1]);
            z->data[ARRAY_IDX_2D(i, j, z->size[1])] = sum;
            assert(ARRAY_IDX_2D(i, j, a->size[1]) < a->size[0]*a->size[1]);
            a->data[ARRAY_IDX_2D(i, j, a->size[1])] = (*activationFunctionPtr)(sum);
        }
    }
}

pearl_tensor *pearl_layer_backward(pearl_layer *layer, pearl_layer *prev_layer, pearl_tensor *dz, pearl_tensor *a, pearl_tensor *z, pearl_tensor *dw, pearl_tensor *db){
    double (*activationFunctionDerivativePtr)(double) = pearl_activation_function_derivative_pointer(prev_layer->activation_function);

    for (int i = 0; i < a->size[0]; i++) {
        for (int j = 0; j < dz->size[0]; j++) {
            double sum_w = 0;
            double sum_b = 0;
            for (int k = 0; k < dz->size[1]; k++) {
                assert(ARRAY_IDX_2D(k, j, dz->size[0]) < dz->size[0]*dz->size[1]);
                assert(ARRAY_IDX_2D(i, k, a->size[1]) < a->size[0]*a->size[1]);
                sum_w += dz->data[ARRAY_IDX_2D(k, j, dz->size[0])] * a->data[ARRAY_IDX_2D(i, k, a->size[1])];
                sum_b += dz->data[ARRAY_IDX_2D(k, j, dz->size[0])];
            }
            assert(ARRAY_IDX_2D(i, j, dw->size[0]) < dw->size[0]*dw->size[1]);
            dw->data[ARRAY_IDX_2D(i, j, dw->size[0])] = sum_w/dz->size[0];
            assert(i < db->size[0]);
            db->data[i] = sum_b/db->size[0];
        }
    }

    pearl_tensor *dz_prev = pearl_tensor_create(2, dz->size[0], layer->weights->size[1]);
    for (int i = 0; i < dz->size[0]; i++) {
        for (int j = 0; j < layer->weights->size[1]; j++) {
            double sum = 0;
            for (int k = 0; k < layer->weights->size[0]; k++) {
                assert(ARRAY_IDX_2D(i, k, dz_prev->size[1]) < dz_prev->size[0]*dz_prev->size[1]);
                assert(ARRAY_IDX_2D(k, j, layer->weights->size[1]) < layer->weights->size[0]*layer->weights->size[1]);
                sum += dz_prev->data[ARRAY_IDX_2D(i, k, dz_prev->size[1])] * layer->weights->data[ARRAY_IDX_2D(k, j, layer->weights->size[1])];
            }
            dz_prev->data[ARRAY_IDX_2D(i, j, dz_prev->size[1])] = sum * (*activationFunctionDerivativePtr)(z->data[ARRAY_IDX_2D(i, j, z->size[1])] );
        }
    }

    return dz_prev;
}
