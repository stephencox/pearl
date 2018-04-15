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
    assert(layer->weights->m == layer->biases->n);
    double (*activationFunctionPtr)(double) = pearl_activation_function_pointer(layer->activation_function);

    for (int i = 0; i < input->m; i++) {
        for (int j = 0; j < layer->weights->m; j++) {
            double sum = 0;
            for (int k = 0; k < layer->weights->n; k++) {
                //printf("input: %d  access: %d\n",input->m*input->n, ARRAY_IDX(i, k, input->n));
                //assert(ARRAY_IDX(i, k, input->n) < input->m*input->n);
                //printf("weights: %d  access: %d\n",layer->weights->m*layer->weights->n, ARRAY_IDX(k, j, layer->weights->m));
                //assert(ARRAY_IDX(k, j, layer->weights->m) < layer->weights->m*layer->weights->n);
                //printf("input=%f  weight=%f\n", input->data[ARRAY_IDX(i, k, input->n)],layer->weights->data[ARRAY_IDX(k, j, layer->weights->m)] );
                sum += input->data[ARRAY_IDX(i, k, input->n)] * layer->weights->data[ARRAY_IDX(k, j, layer->weights->m)];
            }
            sum += layer->biases->data[j];
            z->data[ARRAY_IDX(i, j, z->n)] = sum;
            a->data[ARRAY_IDX(i, j, a->n)] = (*activationFunctionPtr)(sum);
        }
    }
}

void pearl_layer_backward(pearl_layer *layer, pearl_layer *prev_layer, pearl_matrix *dz, pearl_matrix *a, pearl_matrix *z, pearl_matrix *dw, pearl_vector *db, pearl_matrix *dz_prev){
    double (*activationFunctionDerivativePtr)(double) = pearl_activation_function_derivative_pointer(prev_layer->activation_function);

    for (int i = 0; i < dz->m; i++) {
        double sum_b = 0;
        for (int j = 0; j < a->m; j++) {
            double sum_w = 0;
            for (int k = 0; k < a->n; k++) {
                printf("dz: %d  access: %d\n",dz->m*dz->n, ARRAY_IDX(k, j, dz->m));
                assert(ARRAY_IDX(k, j, dz->m) < dz->m*dz->n);
                printf("a: %d  access: %d\n",a->m*a->n, ARRAY_IDX(i, k, a->n));
                assert(ARRAY_IDX(i, k, a->n) < a->m*a->n);
                sum_w += dz->data[ARRAY_IDX(k, j, dz->m)] * a->data[ARRAY_IDX(i, k, a->n)];
                sum_b += dz->data[ARRAY_IDX(k, j, dz->m)];
            }
            sum_w /= dz->m;
            dw->data[ARRAY_IDX(i, j, dw->m)] = sum_w;
        }
        sum_b /= dz->m;
        db->data[i] = sum_b;
    }

    dz_prev = pearl_matrix_create(dz->m, layer->weights->n);
    for (int i = 0; i < dz->m; i++) {
        for (int j = 0; j < layer->weights->n; j++) {
            double sum = 0;
            for (int k = 0; k < layer->weights->m; k++) {
                sum += dz_prev->data[ARRAY_IDX(i, k, dz_prev->n)] * layer->weights->data[ARRAY_IDX(k, j, layer->weights->n)];
            }
            dz_prev->data[ARRAY_IDX(i, j, dz_prev->n)] = sum * (*activationFunctionDerivativePtr)(z->data[ARRAY_IDX(i, j, z->n)] );
        }
    }
}
