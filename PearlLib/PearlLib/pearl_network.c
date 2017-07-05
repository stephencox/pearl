#include <pearl_network.h>

struct pearl_network *pearl_network_create()
{
    struct pearl_network *network = malloc(sizeof(struct pearl_network));
    network->num_layers = 0;
    network->optimiser = pearl_optimiser_sgd;
    return network;
}

void pearl_network_destroy(struct pearl_network *network)
{
    if (network) {
        if (network->layers) {
            free(network->layers);
        }
        free(network);
    }
}

void pearl_network_layer_add(struct pearl_network *network, enum pearl_layer_type type, int neurons, enum pearl_activation_function_type activation_function)
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

void pearl_network_layer_add_input(struct pearl_network *network, int neurons)
{
    pearl_network_layer_add(network, pearl_layer_type_input, neurons, pearl_activation_function_type_linear);
}

void pearl_network_layer_add_output(struct pearl_network *network, int neurons, enum pearl_activation_function_type activation_function)
{
    pearl_network_layer_add(network, pearl_layer_type_input, neurons, activation_function);
}

void pearl_network_layer_add_dropout(struct pearl_network *network, int neurons, enum pearl_activation_function_type activation_function, double dropout_rate)
{
    pearl_network_layer_add(network, pearl_layer_type_dropout, neurons, activation_function);
    network->layers[network->num_layers - 1].dropout_rate = dropout_rate;
}
