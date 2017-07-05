#ifndef PEARL_NETWORK_H
#define PEARL_NETWORK_H

#include <stdlib.h>
#include <pearl_layer.h>
#include <pearl_optimiser.h>

struct pearl_network {
    int num_layers;
    struct pearl_layer *layers;
    enum pearl_optimiser optimiser;
    int *neuron_weights;
};

struct pearl_network *pearl_network_create();
void pearl_network_destroy(struct pearl_network *network);
void pearl_network_layer_add(struct pearl_network *network, enum pearl_layer_type type, int neurons, enum pearl_activation_function_type activation_function);
void pearl_network_layer_add_input(struct pearl_network *network, int neurons);
void pearl_network_layer_add_output(struct pearl_network *network, int neurons, enum pearl_activation_function_type activation_function);
void pearl_network_layer_add_dropout(struct pearl_network *network, int neurons, enum pearl_activation_function_type activation_function, double dropout_rate);

#endif // PEARL_NETWORK_H
