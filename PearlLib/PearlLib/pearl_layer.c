#include <pearl_layer.h>

void pearl_layer_initialise(pearl_layer **layer, const int num_neurons_next_layer)
{
    pearl_layer *layer_p = (*layer);
    if (layer_p) {
        if (layer_p->biases == NULL) {
            layer_p->biases = pearl_tensor_create(1, layer_p->neurons);
        }
        if (layer_p->weights == NULL) {
            layer_p->weights = pearl_tensor_create(2, num_neurons_next_layer, layer_p->neurons);
            //Glorot, X. & Bengio, Y.. (2010). Understanding the difficulty of training deep feedforward neural networks. Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, in PMLR 9:249-256
            double scale = sqrt(6.0 / (layer_p->neurons + num_neurons_next_layer));
            for (unsigned int i = 0; i < layer_p->weights->size[0] * layer_p->weights->size[1]; i++) {
                layer_p->weights->data[i] = -1.0 + ((float)rand() / (float)(RAND_MAX)) * scale * 2.0;
            }
        }
    }
}

void pearl_layer_destroy(pearl_layer **layer)
{
    pearl_layer *layer_p = (*layer);
    if (layer_p) {
        pearl_tensor_destroy(&(layer_p->biases));
        if (layer_p->weights) {
            pearl_tensor_destroy(&layer_p->weights);
        }
        free(layer);
        layer = NULL;
    }
}

void pearl_layer_print(const pearl_layer *layer)
{
    if (layer) {
        printf("Type: pearl_layer\n");
        printf("Type: ");
        switch (layer->type) {
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

void pearl_layer_forward(pearl_layer **layer, const pearl_tensor *input, pearl_tensor **z, pearl_tensor **a)
{
    pearl_layer *layer_p = (*layer);
    pearl_tensor *z_p = (*z);
    pearl_tensor *a_p = (*a);
    assert(layer_p->weights->size[1] == input->size[0]);
    assert(layer_p->weights->dimension == 2);
    assert(layer_p->biases->dimension == 1);
    assert(layer_p->weights->size[0] == layer_p->biases->size[0]);
    double (*activationFunctionPtr)(double) = pearl_activation_function_pointer(layer_p->activation_function);

    for (unsigned int i = 0; i < layer_p->weights->size[0]; i++) {
        for (unsigned int j = 0; j < input->size[1]; j++) {
            double sum = 0.0;
            for (unsigned int k = 0; k < layer_p->weights->size[1]; k++) {
                assert(ARRAY_IDX_2D(i, k, layer_p->weights->size[1]) < layer_p->weights->size[0]*layer_p->weights->size[1]);
                assert(ARRAY_IDX_2D(k, j, input->size[1]) < input->size[0]*input->size[1]);
                sum += layer_p->weights->data[ARRAY_IDX_2D(i, k, layer_p->weights->size[1])] * input->data[ARRAY_IDX_2D(k, j, input->size[1])];
            }
            sum += layer_p->biases->data[i];
            assert(ARRAY_IDX_2D(i, j, z_p->size[1]) < z_p->size[0]*z_p->size[1]);
            z_p->data[ARRAY_IDX_2D(i, j, z_p->size[1])] = sum;
            assert(ARRAY_IDX_2D(i, j, a_p->size[1]) < a_p->size[0]*a_p->size[1]);
            a_p->data[ARRAY_IDX_2D(i, j, a_p->size[1])] = (*activationFunctionPtr)(sum);
        }
    }
}

void pearl_layer_backward(pearl_layer *layer, pearl_layer *prev_layer, pearl_tensor *dz, pearl_tensor *a, pearl_tensor *z, pearl_tensor *dw, pearl_tensor *db, pearl_tensor **dz_prev)
{
    pearl_layer_backward_weights_biases(dz, a, &dw, &db);
    pearl_layer_backward_activation(layer, prev_layer, dz, z, dz_prev);
}

void pearl_layer_backward_weights_biases(const pearl_tensor *dz, const pearl_tensor *a, pearl_tensor **dw, pearl_tensor **db)
{
    pearl_tensor *dw_p = (*dw);
    pearl_tensor *db_p = (*db);
    assert(dz->dimension == 2);
    assert(a->dimension == 2);
    assert(dz->size[1] == a->size[1]);
    assert(dw_p->dimension == 2);
    assert(dw_p->size[0] == dz->size[0]);
    assert(dw_p->size[1] == a->size[0]);
    assert(db_p->dimension == 1);
    for (unsigned int i = 0; i < dz->size[0]; i++) {
        double sum_b = 0;
        for (unsigned int j = 0; j < a->size[0]; j++) {
            double sum_w = 0;
            for (unsigned int k = 0; k < dz->size[1]; k++) {
                assert(ARRAY_IDX_2D(i, k, a->size[1]) < a->size[0]*a->size[1]);
                assert(ARRAY_IDX_2D(j, k, a->size[0]) < a->size[0]*a->size[1]);
                sum_w += dz->data[ARRAY_IDX_2D(i, k, dz->size[1])] * a->data[ARRAY_IDX_2D(j, k, a->size[0])];
                sum_b += dz->data[ARRAY_IDX_2D(i, k, dz->size[1])] / db_p->size[0];
            }
            assert(ARRAY_IDX_2D(i, j, dw_p->size[1]) < dw_p->size[0]*dw_p->size[1]);
            dw_p->data[ARRAY_IDX_2D(i, j, dw_p->size[1])] = sum_w / dz->size[0];
        }
        assert(i < db_p->size[0]);
        db_p->data[i] = sum_b;
    }
}


void pearl_layer_backward_activation(const pearl_layer *layer, const pearl_layer *prev_layer, const pearl_tensor *dz, const pearl_tensor *z, pearl_tensor **dz_prev)
{
    pearl_tensor *dz_prev_p = (*dz_prev);
    double (*activationFunctionDerivativePtr)(double) = pearl_activation_function_derivative_pointer(prev_layer->activation_function);
    for (unsigned int i = 0; i < layer->weights->size[1]; i++) {
        for (unsigned int j = 0; j < dz->size[1]; j++) {
            double sum = 0;
            for (unsigned int k = 0; k < layer->weights->size[0]; k++) {
                assert(ARRAY_IDX_2D(i, k, dz->size[1]) < dz->size[0]*dz->size[1]);
                assert(ARRAY_IDX_2D(j, k, layer->weights->size[0]) < layer->weights->size[0]*layer->weights->size[1]);
                sum += layer->weights->data[ARRAY_IDX_2D(j, k, layer->weights->size[0])] * dz->data[ARRAY_IDX_2D(i, k, dz->size[1])];
            }
            assert(ARRAY_IDX_2D(i, j, z->size[1]) < z->size[0]*z->size[1]);
            assert(ARRAY_IDX_2D(i, j, dz_prev_p->size[1]) < dz_prev_p->size[0]*dz_prev_p->size[1]);
            dz_prev_p->data[ARRAY_IDX_2D(i, j, dz_prev_p->size[1])] = sum * (*activationFunctionDerivativePtr)(z->data[ARRAY_IDX_2D(i, j, z->size[1])]);
        }
    }
}


void pearl_layer_update(pearl_layer *layer, pearl_tensor *dw, pearl_tensor *db, double learning_rate)
{
    printf("W\n");
    pearl_tensor_print(layer->weights);
    printf("b\n");
    pearl_tensor_print(layer->biases);
    printf("dW\n");
    pearl_tensor_print(dw);
    printf("db\n");
    pearl_tensor_print(db);

    assert(layer->weights->dimension == 2);
    assert(layer->weights->size[0] == dw->size[0]);
    assert(layer->weights->size[1] == dw->size[1]);
    pearl_tensor_print(layer->weights);
    for (unsigned int i = 0; i < layer->weights->size[0]; i++) {
        for (unsigned int j = 0; j < layer->weights->size[1]; j++) {
            layer->weights->data[ARRAY_IDX_2D(i, j, layer->weights->size[1])] -= learning_rate * dw->data[ARRAY_IDX_2D(i, j, dw->size[1])];
        }
    }
    pearl_tensor_print(layer->weights);
}
