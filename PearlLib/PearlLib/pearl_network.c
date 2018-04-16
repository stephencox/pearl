#include <pearl_network.h>

// TODO: Save network in sqlite

PEARL_API pearl_network *pearl_network_create()
{
    pearl_network *network = malloc(sizeof(pearl_network));
    network->num_layers = 0;
    network->optimiser = pearl_optimiser_sgd;
    network->learning_rate = 1e-3;
    return network;
}

PEARL_API void pearl_network_destroy(pearl_network *network)
{
    if (network) {
        if (network->layers) {
            for (int i = 0; i < network->num_layers; i++) {
                pearl_layer_destroy(&network->layers[i]);
            }
            free(network->layers);
        }
        free(network);
    }
}

PEARL_API void pearl_network_layer_add(pearl_network *network, enum pearl_layer_type type, int neurons, enum pearl_activation_function_type activation_function)
{
    network->num_layers++;
    if (network->num_layers > 1) {
        network->layers = (pearl_layer *)realloc(network->layers, network->num_layers * sizeof(pearl_layer)); //TODO: error checking
    }
    else {
        network->layers = (pearl_layer *)malloc(sizeof(pearl_layer));
    }
    network->layers[network->num_layers - 1].type = type;
    network->layers[network->num_layers - 1].neurons = neurons;
    network->layers[network->num_layers - 1].activation_function = activation_function;
    network->layers[network->num_layers - 1].weights = NULL;
    network->layers[network->num_layers - 1].biases = NULL;
}

PEARL_API void pearl_network_layer_add_input(pearl_network *network, int neurons)
{
    pearl_network_layer_add(network, pearl_layer_type_input, neurons, pearl_activation_function_type_linear);
}

PEARL_API void pearl_network_layer_add_output(pearl_network *network, int neurons, enum pearl_activation_function_type activation_function)
{
    pearl_network_layer_add(network, pearl_layer_type_output, neurons, activation_function);
}

PEARL_API void pearl_network_layer_add_dropout(pearl_network *network, int neurons, enum pearl_activation_function_type activation_function, double dropout_rate)
{
    pearl_network_layer_add(network, pearl_layer_type_dropout, neurons, activation_function);
    network->layers[network->num_layers - 1].dropout_rate = dropout_rate;
}

PEARL_API void pearl_network_layer_add_fully_connect(pearl_network *network, int neurons, enum pearl_activation_function_type activation_function)
{
    pearl_network_layer_add(network, pearl_layer_type_fully_connect, neurons, activation_function);
}

PEARL_API void pearl_network_layers_initialise(pearl_network *network)
{
    srand((unsigned int)time(NULL));
    for (int i = 1; i < network->num_layers; i++) {
        pearl_layer_initialise(&network->layers[i], &network->layers[i - 1]);
    }
}

PEARL_API void pearl_network_train_epoch(pearl_network *network, const pearl_tensor *input, const pearl_tensor *output)
{
    pearl_tensor **z = calloc(network->num_layers - 1, sizeof(pearl_tensor *));
    pearl_tensor **a = calloc(network->num_layers - 1, sizeof(pearl_tensor *));
    // Forward
    a[0] = pearl_tensor_copy(input);
    for (int i = 1; i < network->num_layers-1; i++) {
        assert(z[i] == NULL);
        z[i] = pearl_tensor_create(2, input->size[0], network->layers[i].weights->size[0]);
        assert(a[i] == NULL);
        a[i] = pearl_tensor_create(2, input->size[0], network->layers[i].weights->size[0]);
        pearl_layer_forward(&network->layers[i], a[i-1], z[i], a[i]);
    }
    // Cost
    double cost = 0.0;
    pearl_tensor *al = a[network->num_layers - 2];
    for (int i = 0; i < output->size[0]; i++) {
        if (output->data[i] > 0.0) {
            cost += log(al->data[i]);
        }
        else {
            cost += log(1.0 - al->data[i]);
        }
    }
    cost /= (double)(-output->size[1]);
    printf("Loss: %f\n", cost);

    //Backward
    pearl_tensor **dw = calloc(network->num_layers - 1, sizeof(pearl_tensor *));
    pearl_tensor **db = calloc(network->num_layers - 1, sizeof(pearl_tensor *));
    pearl_tensor **dz = calloc(network->num_layers - 1, sizeof(pearl_tensor *));
    for (int i = network->num_layers - 1; i > 1; i--) {
        if (i == network->num_layers - 1) {
            assert(dz[i - 1] == NULL);
            dz[i - 1] = pearl_tensor_create(2, output->size[0], output->size[1]);
            for (int j = 0; j < output->size[0]; j++) {
                if (output->data[j] > 0.0) {
                    dz[i - 1]->data[j] = - (output->data[j] / al->data[j]);
                }
                else {
                    dz[i - 1]->data[j] = - ((1.0 - output->data[j]) / (1.0 - al->data[j]));
                }
            }
            assert(dw[i - 1] == NULL);
            dw[i - 1] = pearl_tensor_create(2, dz[i-1]->size[0], a[i-1]->size[0]);
            assert(db[i - 1] == NULL);
            db[i - 1] = pearl_tensor_create(1, a[i-1]->size[0]);
            assert(dz[i - 2] == NULL);
            dz[i - 2] = pearl_layer_backward(&network->layers[i], &network->layers[i-1], dz[i - 1], a[i - 1], z[i - 1], dw[i - 1], db[i - 1]);
        }
        else {
            assert(dw[i - 1] == NULL);
            dw[i - 1] = pearl_tensor_create(2, input->size[0], network->layers[i].weights->size[0]);
            assert(db[i - 1] == NULL);
            db[i - 1] = pearl_tensor_create(1, input->size[0]);
            assert(dz[i - 2] == NULL);
            dz[i - 2] = pearl_layer_backward(&network->layers[i], &network->layers[i-1], dz[i - 1], a[i - 1], z[i - 1], dw[i - 1], db[i - 1]);
        }
    }
    // Clean
    for (int i = 0; i < network->num_layers - 1; i++) {
        pearl_tensor_destroy(a[i]);
        pearl_tensor_destroy(z[i]);
        pearl_tensor_destroy(dw[i]);
        pearl_tensor_destroy(db[i]);
        pearl_tensor_destroy(dz[i]);
    }
    free(a);
    free(z);
    free(dw);
    free(db);
    free(dz);
}
