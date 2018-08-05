#include <pearl_network.h>

PEARL_API pearl_network *pearl_network_create(const unsigned int num_input, const unsigned int num_output)
{
    srand((unsigned int)time(NULL));
    pearl_network *network = malloc(sizeof(pearl_network));
    network->num_layers = 0;
    network->optimiser = pearl_optimiser_sgd;
    network->loss_type = pearl_loss_binary_cross_entropy;
    network->learning_rate = 1e-3;
    network->num_input = num_input;
    network->num_output = num_output;
    network->layers = NULL;
    network->version.major = PEARL_NETWORK_VERSION_MAJOR;
    network->version.minor = PEARL_NETWORK_VERSION_MINOR;
    network->version.revision = PEARL_NETWORK_VERSION_REVISION;
    return network;
}

PEARL_API void pearl_network_destroy(pearl_network **network)
{
    if (*network != NULL) {
        if ((*network)->layers != NULL) {
            for (unsigned int i = 0; i < (*network)->num_layers; i++) {
                pearl_layer_destroy(&(*network)->layers[i]);
            }
            free((*network)->layers);
            (*network)->layers = NULL;
        }
        free(*network);
        *network = NULL;
    }
}

PEARL_API void pearl_network_layer_add(pearl_network **network, const pearl_layer_type type, const int neurons, const pearl_activation_function_type activation_function)
{
    (*network)->num_layers++;
    if ((*network)->num_layers > 1) {
        (*network)->layers = realloc((*network)->layers, (*network)->num_layers * sizeof(pearl_layer *)); //TODO: error checking
    }
    else {
        (*network)->layers = calloc(1, sizeof(pearl_layer *));
    }
    pearl_layer *layer = malloc(sizeof(pearl_layer));
    layer->type = type;
    layer->neurons = neurons;
    layer->activation_function = activation_function;
    layer->weights = NULL;
    layer->biases = NULL;
    //layer->dropout_rate = 0.0;
    layer->version.major = PEARL_LAYER_VERSION_MAJOR;
    layer->version.minor = PEARL_LAYER_VERSION_MINOR;
    layer->version.revision = PEARL_LAYER_VERSION_REVISION;
    (*network)->layers[(*network)->num_layers - 1] = layer;
}

PEARL_API void pearl_network_layer_add_output(pearl_network **network, const pearl_activation_function_type activation_function)
{
    pearl_network_layer_add(network, pearl_layer_type_fully_connect, (*network)->num_output, activation_function);
}

//PEARL_API void pearl_network_layer_add_dropout(pearl_network **network, const int neurons, const pearl_activation_function_type activation_function, const double dropout_rate)
//{
//    pearl_network_layer_add(network, pearl_layer_type_dropout, neurons, activation_function);
//    pearl_network *network_p = (*network);
//    network_p->layers[network_p->num_layers - 1]->dropout_rate = dropout_rate;
//}

PEARL_API void pearl_network_layer_add_fully_connect(pearl_network **network, const int neurons, const pearl_activation_function_type activation_function)
{
    pearl_network_layer_add(network, pearl_layer_type_fully_connect, neurons, activation_function);
}

PEARL_API void pearl_network_layers_initialise(pearl_network **network)
{
    if ((*network)->layers != NULL) {
        for (unsigned int i = 0; i < (*network)->num_layers; i++) {
            unsigned int num_neurons_prev_layer = (i == 0 ? (*network)->num_input : (*network)->layers[i - 1]->neurons);
            pearl_layer_initialise(&(*network)->layers[i], num_neurons_prev_layer);
        }
    }
}

PEARL_API double pearl_network_train_epoch(pearl_network **network, const pearl_tensor *input, const pearl_tensor *output)
{
    // Forward
    pearl_tensor **z = calloc((*network)->num_layers, sizeof(pearl_tensor *));
    pearl_tensor **a = calloc((*network)->num_layers + 1, sizeof(pearl_tensor *));
    pearl_network_forward(network, input, z, a);

    // Cost
    pearl_tensor *al = a[(*network)->num_layers];
    double cost = 0.0;
    switch ((*network)->loss_type) {
        case pearl_loss_binary_cross_entropy:
            cost = pearl_loss_binary_cross_entropy_cost(output, al);
            break;
        case pearl_loss_mean_squared_error:
            cost = pearl_loss_binary_cross_entropy_cost(output, al);
            break;
    }
    //Backward
    pearl_tensor **dw = calloc((*network)->num_layers, sizeof(pearl_tensor *));
    pearl_tensor **db = calloc((*network)->num_layers, sizeof(pearl_tensor *));
    pearl_tensor **dZ = calloc((*network)->num_layers, sizeof(pearl_tensor *));
    pearl_tensor **dA = calloc((*network)->num_layers + 1, sizeof(pearl_tensor *));
    unsigned int i;
    for (unsigned int num_layer = (*network)->num_layers; num_layer > 0; num_layer--) {
        i = num_layer - 1;
        if (i == (*network)->num_layers - 1) {
            assert(dA[i + 1] == NULL);
            dA[i + 1] = pearl_tensor_create(2, output->size[0], output->size[1]);
            for (unsigned int j = 0; j < output->size[1]; j++) {
                for (unsigned int x = 0; x < output->size[0]; x++) {
                    assert(ARRAY_IDX_2D(j, x, output->size[0]) < output->size[0]*output->size[1]);
                    assert(ARRAY_IDX_2D(x, j, dA[i + 1]->size[1]) < dA[i + 1]->size[0]*dA[i + 1]->size[1]);
                    assert(ARRAY_IDX_2D(x, j, al->size[1]) < al->size[0]*al->size[1]);
                    dA[i + 1]->data[ARRAY_IDX_2D(x, j, dA[i + 1]->size[1])] = - (output->data[ARRAY_IDX_2D(j, x, output->size[0])] / al->data[ARRAY_IDX_2D(x, j, al->size[1])] - (1.0 - output->data[ARRAY_IDX_2D(j, x, output->size[0])]) / (1.0 - al->data[ARRAY_IDX_2D(x, j, al->size[1])]));
                }
            }
        }

        assert(dw[i] == NULL);
        assert(dA[i + 1] != NULL);
        dw[i] = pearl_tensor_create(2, dA[i + 1]->size[0], a[i]->size[0]);
        assert(db[i] == NULL);
        db[i] = pearl_tensor_create(1, dA[i + 1]->size[0]);
        assert(dZ[i] == NULL);
        dZ[i] = pearl_tensor_create(2, dA[i + 1]->size[0], dA[i + 1]->size[1]);
        assert(dA[i] == NULL);
        dA[i] = pearl_tensor_create(2, (*network)->layers[i]->weights->size[1], dZ[i]->size[1]);
        double (*activationFunctionDerivativePtr)(double) = pearl_activation_function_derivative_pointer((*network)->layers[i]->activation_function);

        for (unsigned int j = 0; j < dZ[i]->size[1]; j++) {
            for (unsigned int x = 0; x < dZ[i]->size[0]; x++) {
                assert(ARRAY_IDX_2D(j, x, z[i]->size[0]) < z[i]->size[0]*z[i]->size[1]);
                assert(ARRAY_IDX_2D(x, j, dZ[i]->size[1]) < dZ[i]->size[0]*dZ[i]->size[1]);
                assert(ARRAY_IDX_2D(x, j, dA[i + 1]->size[1]) < dA[i + 1]->size[0]*dA[i + 1]->size[1]);
                dZ[i]->data[ARRAY_IDX_2D(x, j, dZ[i]->size[1])] = activationFunctionDerivativePtr(z[i]->data[ARRAY_IDX_2D(x, j, z[i]->size[1])]) * dA[i + 1]->data[ARRAY_IDX_2D(x, j, dA[i + 1]->size[1])];
            }
        }
        pearl_layer_backward((*network)->layers[i], dZ[i], a[i], &dw[i], &db[i], &dA[i]);
    }
    //Update
    for (unsigned int i = 0; i < (*network)->num_layers; i++) {
        pearl_layer_update((*network)->layers[i], dw[i], db[i], (*network)->learning_rate);
    }
    // Clean
    for (unsigned int i = 0; i < (*network)->num_layers; i++) {
        pearl_tensor_destroy(&a[i]);
        pearl_tensor_destroy(&z[i]);
        pearl_tensor_destroy(&dw[i]);
        pearl_tensor_destroy(&db[i]);
        pearl_tensor_destroy(&dA[i]);
        pearl_tensor_destroy(&dZ[i]);
    }
    pearl_tensor_destroy(&a[(*network)->num_layers]);
    pearl_tensor_destroy(&dA[(*network)->num_layers]);
    free(a);
    free(z);
    free(dw);
    free(db);
    free(dA);
    free(dZ);

    return cost;
}

void pearl_network_forward(pearl_network **network, const pearl_tensor *input, pearl_tensor **z, pearl_tensor **a)
{
    a[0] = pearl_tensor_create(2, input->size[1], input->size[0]);
    for (unsigned int i = 0; i < input->size[0]; i++) {
        for (unsigned int j = 0; j < input->size[1]; j++) {
            a[0]->data[ARRAY_IDX_2D(j, i, a[0]->size[1])] = input->data[ARRAY_IDX_2D(i, j, input->size[1])];
        }
    }

    for (unsigned int i = 0; i < (*network)->num_layers; i++) {
        assert(z[i] == NULL);
        z[i] = pearl_tensor_create(2, (*network)->layers[i]->weights->size[0], a[i]->size[1]);
        assert(a[i + 1] == NULL);
        a[i + 1] = pearl_tensor_create(2, (*network)->layers[i]->weights->size[0], a[i]->size[1]);
        pearl_layer_forward(&(*network)->layers[i], a[i], &z[i], &a[i + 1]);
    }
}

PEARL_API pearl_tensor *pearl_network_calculate(pearl_network **network, const pearl_tensor *input)
{
    pearl_tensor **z = calloc((*network)->num_layers, sizeof(pearl_tensor *));
    pearl_tensor **a = calloc((*network)->num_layers + 1, sizeof(pearl_tensor *));
    pearl_network_forward(network, input, z, a);
    pearl_tensor *output = pearl_tensor_copy(a[(*network)->num_layers]);
    for (unsigned int i = 0; i < (*network)->num_layers; i++) {
        pearl_tensor_destroy(&a[i]);
        pearl_tensor_destroy(&z[i]);
    }
    pearl_tensor_destroy(&a[(*network)->num_layers]);
    free(a);
    free(z);
    return output;
}
