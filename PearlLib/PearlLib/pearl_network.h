#ifndef PEARL_NETWORK_H
#define PEARL_NETWORK_H

#include <stdlib.h>
#include <stdio.h>
#include <pearl_global.h>
#include <pearl_layer.h>
#include <pearl_optimiser.h>
#include <pearl_tensor.h>
#include <pearl_loss.h>
#include <pearl_version.h>
#include <time.h>

#define PEARL_NETWORK_VERSION_MAJOR 1
#define PEARL_NETWORK_VERSION_MINOR 0
#define PEARL_NETWORK_VERSION_REVISION 0

typedef struct {
    int num_layers;
    pearl_layer *layers;
    pearl_optimiser optimiser;
    pearl_loss loss;
    double learning_rate;
    unsigned int num_input;
    unsigned int num_output;
    pearl_version version;
} pearl_network;

PEARL_API pearl_network *pearl_network_create(unsigned int num_input, unsigned int num_output);
PEARL_API void pearl_network_destroy(pearl_network *network);
PEARL_API void pearl_network_save(char *filename, pearl_network *network);
PEARL_API void pearl_network_layer_add(pearl_network *network, pearl_layer_type type, int neurons, pearl_activation_function_type activation_function);
PEARL_API void pearl_network_layer_add_output(pearl_network *network, int neurons, pearl_activation_function_type activation_function);
PEARL_API void pearl_network_layer_add_fully_connect(pearl_network *network, int neurons, pearl_activation_function_type activation_function);
PEARL_API void pearl_network_layer_add_dropout(pearl_network *network, int neurons, pearl_activation_function_type activation_function, double dropout_rate);
PEARL_API void pearl_network_layers_initialise(pearl_network *network);
PEARL_API void pearl_network_train_epoch(pearl_network *network, const pearl_tensor *input, const pearl_tensor *output);

#endif // PEARL_NETWORK_H
