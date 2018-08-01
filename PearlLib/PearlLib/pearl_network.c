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

PEARL_API void pearl_network_train_epoch(pearl_network **network, const pearl_tensor *input, const pearl_tensor *output)
{
    pearl_tensor **z = calloc((*network)->num_layers, sizeof(pearl_tensor *));
    pearl_tensor **a = calloc((*network)->num_layers + 1, sizeof(pearl_tensor *));
    // Forward
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
    // Cost
    pearl_tensor *al = a[(*network)->num_layers];
    double cost = 0.0;
    switch ((*network)->loss_type) {
        case pearl_loss_binary_cross_entropy:
            cost = pearl_loss_binary_cross_entropy_cost(output, al);
            break;
        default:
            printf("Invalid loss function\n");
            break;
    }
    //Backward
    pearl_tensor **dw = calloc((*network)->num_layers, sizeof(pearl_tensor *));
    pearl_tensor **db = calloc((*network)->num_layers, sizeof(pearl_tensor *));
    pearl_tensor **da = calloc((*network)->num_layers, sizeof(pearl_tensor *));
    pearl_tensor **dz = calloc((*network)->num_layers + 1, sizeof(pearl_tensor *));
    for (int i = (*network)->num_layers - 1; i >= 0; i--) {
        if (i == (*network)->num_layers - 1) {
            assert(da[i] == NULL);
            assert(dz[i + 1] == NULL);
            da[i] = pearl_tensor_create(2, output->size[0], output->size[1]);
            dz[i + 1] = pearl_tensor_create(2, output->size[0], output->size[1]);
            double (*activationFunctionDerivativePtr)(double) = pearl_activation_function_derivative_pointer((*network)->layers[i]->activation_function);
            for (unsigned int j = 0; j < output->size[1]; j++) {
                for (unsigned int x = 0; x < output->size[0]; x++) {
                    assert(ARRAY_IDX_2D(j, x, output->size[0]) < output->size[0]*output->size[1]);
                    assert(ARRAY_IDX_2D(x, j, da[i]->size[1]) < da[i]->size[0]*da[i]->size[1]);
                    assert(ARRAY_IDX_2D(x, j, dz[i + 1]->size[1]) < dz[i + 1]->size[0]*dz[i + 1]->size[1]);
                    assert(ARRAY_IDX_2D(x, j, al->size[1]) < al->size[0]*al->size[1]);
                    double calc = - (output->data[ARRAY_IDX_2D(j, x, output->size[0])] / al->data[ARRAY_IDX_2D(x, j, al->size[1])] - (1.0 - output->data[ARRAY_IDX_2D(j, x, output->size[0])]) / (1.0 - al->data[ARRAY_IDX_2D(x, j, al->size[1])]));
                    da[i]->data[ARRAY_IDX_2D(x, j, da[i]->size[1])] = calc;
                    dz[i + 1]->data[ARRAY_IDX_2D(x, j, dz[i + 1]->size[1])] = activationFunctionDerivativePtr(z[i]->data[ARRAY_IDX_2D(x, j, z[i]->size[1])]) * calc;
                }
            }
        }

        assert(dw[i] == NULL);
        dw[i] = pearl_tensor_create(2, dz[i + 1]->size[0], a[i]->size[0]);
        assert(db[i] == NULL);
        db[i] = pearl_tensor_create(1, dz[i + 1]->size[0]);
        assert(dz[i] == NULL);
        dz[i] = pearl_tensor_create(2, (*network)->layers[i]->weights->size[1], dz[i + 1]->size[1]);
        pearl_layer_backward((*network)->layers[i], dz[i + 1], a[i], &dw[i], &db[i], &dz[i]);
    }
    //Update
    for (unsigned int i = 0; i < (*network)->num_layers; i++) {
        pearl_layer_update((*network)->layers[i], dw[i], db[i], (*network)->learning_rate);
    }
    // Clean
    for (unsigned int i = 0; i < (*network)->num_layers - 1; i++) {
        pearl_tensor_destroy(&a[i]);
        pearl_tensor_destroy(&z[i]);
        pearl_tensor_destroy(&dw[i]);
        pearl_tensor_destroy(&db[i]);
        pearl_tensor_destroy(&dz[i]);
    }
    free(a);
    free(z);
    free(dw);
    free(db);
    free(dz);
}
