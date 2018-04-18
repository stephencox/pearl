#ifndef PEARL_NETWORK_H
#define PEARL_NETWORK_H

#include <stdlib.h>
#include <pearl_global.h>
#include <pearl_layer.h>
#include <pearl_optimiser.h>
#include <pearl_tensor.h>
#include <time.h>

typedef struct {
    int num_layers;
    pearl_layer *layers;
    enum pearl_optimiser optimiser;
    double learning_rate;
    int num_input;
    int num_output;
} pearl_network;

PEARL_API pearl_network *pearl_network_create(int num_input, int num_output);
PEARL_API void pearl_network_destroy(pearl_network *network);
PEARL_API void pearl_network_layer_add(pearl_network *network, enum pearl_layer_type type, int neurons, enum pearl_activation_function_type activation_function);
PEARL_API void pearl_network_layer_add_output(pearl_network *network, int neurons, enum pearl_activation_function_type activation_function);
PEARL_API void pearl_network_layer_add_fully_connect(pearl_network *network, int neurons, enum pearl_activation_function_type activation_function);
PEARL_API void pearl_network_layer_add_dropout(pearl_network *network, int neurons, enum pearl_activation_function_type activation_function, double dropout_rate);
PEARL_API void pearl_network_layers_initialise(pearl_network *network);
PEARL_API void pearl_network_train_epoch(pearl_network *network, const pearl_tensor *input, const pearl_tensor *output);

#endif // PEARL_NETWORK_H
