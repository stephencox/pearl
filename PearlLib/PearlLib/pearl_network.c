#include <pearl_network.h>

PEARL_API pearl_network *pearl_network_create()
{
    pearl_network *network = malloc(sizeof(pearl_network));
    network->num_layers = 0;
    network->optimiser = pearl_optimiser_sgd;
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
    pearl_network_layer_add(network, pearl_layer_type_input, neurons, activation_function);
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
    for (int i = 1; i < network->num_layers; i++) {
        pearl_layer_initialise(&network->layers[i], &network->layers[i - 1]);
    }
}

PEARL_API void pearl_network_train_epoch(pearl_network *network, const pearl_matrix *input, const pearl_matrix *output)
{
    for (int i = 1; i < network->num_layers; i++) {
        pearl_matrix *z = pearl_layer_forward(&network->layers[i], input);
    }
}
