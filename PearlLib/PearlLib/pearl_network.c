#include <pearl_network.h>

PEARL_API struct pearl_network *pearl_network_create()
{
    struct pearl_network *network = malloc(sizeof(struct pearl_network));
    network->num_layers = 0;
    network->optimiser = pearl_optimiser_sgd;
    return network;
}

PEARL_API void pearl_network_destroy(struct pearl_network *network)
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

PEARL_API void pearl_network_layer_add(struct pearl_network *network, enum pearl_layer_type type, int neurons, enum pearl_activation_function_type activation_function)
{
    network->num_layers++;
    if (network->num_layers > 0) {
        network->layers = (struct pearl_layer *)realloc(network->layers, network->num_layers * sizeof(struct pearl_layer)); //TODO: error checking
    }
    else {
        network->layers = (struct pearl_layer *)malloc(sizeof(struct pearl_layer));
    }
    network->layers[network->num_layers - 1].type = type;
    network->layers[network->num_layers - 1].neurons = neurons;
    network->layers[network->num_layers - 1].activation_function = activation_function;
}

PEARL_API void pearl_network_layer_add_input(struct pearl_network *network, int neurons)
{
    pearl_network_layer_add(network, pearl_layer_type_input, neurons, pearl_activation_function_type_linear);
}

PEARL_API void pearl_network_layer_add_output(struct pearl_network *network, int neurons, enum pearl_activation_function_type activation_function)
{
    pearl_network_layer_add(network, pearl_layer_type_input, neurons, activation_function);
}

PEARL_API void pearl_network_layer_add_dropout(struct pearl_network *network, int neurons, enum pearl_activation_function_type activation_function, double dropout_rate)
{
    pearl_network_layer_add(network, pearl_layer_type_dropout, neurons, activation_function);
    network->layers[network->num_layers - 1].dropout_rate = dropout_rate;
}

PEARL_API void pearl_network_layer_add_fully_connect(struct pearl_network *network, int neurons, enum pearl_activation_function_type activation_function)
{
    pearl_network_layer_add(network, pearl_layer_type_fully_connect, neurons, activation_function);
}

PEARL_API void pearl_network_layers_initialise(struct pearl_network *network)
{
    for (int i = 1; i < network->num_layers; i++) {
        pearl_layer_initialise(&network->layers[i], &network->layers[i - 1]);
    }
}

PEARL_API void pearl_network_train_epoch(struct pearl_network *network, const double *input, const double *output)
{
    //Forward
    for (int i = 1; i < network->num_layers; i++) {
    }
}
