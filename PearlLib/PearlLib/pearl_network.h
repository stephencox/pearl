#ifndef PEARL_NETWORK_H
#define PEARL_NETWORK_H

#include <stdlib.h>
#include <pearl_global.h>
#include <pearl_layer.h>
#include <pearl_optimiser.h>
#include <pearl_matrix.h>

struct pearl_network {
    int num_layers;
    struct pearl_layer *layers;
    enum pearl_optimiser optimiser;
};

PEARL_API struct pearl_network *pearl_network_create();
PEARL_API void pearl_network_destroy(struct pearl_network *network);
PEARL_API void pearl_network_layer_add(struct pearl_network *network, enum pearl_layer_type type, int neurons, enum pearl_activation_function_type activation_function);
PEARL_API void pearl_network_layer_add_input(struct pearl_network *network, int neurons);
PEARL_API void pearl_network_layer_add_output(struct pearl_network *network, int neurons, enum pearl_activation_function_type activation_function);
PEARL_API void pearl_network_layer_add_fully_connect(struct pearl_network *network, int neurons, enum pearl_activation_function_type activation_function);
PEARL_API void pearl_network_layer_add_dropout(struct pearl_network *network, int neurons, enum pearl_activation_function_type activation_function, double dropout_rate);
PEARL_API void pearl_network_layers_initialise(struct pearl_network *network);
PEARL_API void pearl_network_train_epoch(struct pearl_network *network, const struct pearl_matrix *input, const struct pearl_matrix *output);

#endif // PEARL_NETWORK_H
