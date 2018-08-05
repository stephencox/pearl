#ifndef PEARL_NETWORK_H
#define PEARL_NETWORK_H

#include <stdlib.h>
#include <pearl_global.h>
#include <pearl_layer.h>
#include <pearl_optimiser.h>
#include <pearl_tensor.h>
#include <pearl_loss.h>
#include <pearl_version.h>
#include <math.h>
#include <time.h>

#define PEARL_NETWORK_VERSION_MAJOR 1
#define PEARL_NETWORK_VERSION_MINOR 0
#define PEARL_NETWORK_VERSION_REVISION 0

typedef struct {
    unsigned int num_layers;
    pearl_layer **layers;
    pearl_optimiser optimiser;
    pearl_loss loss_type;
    double learning_rate;
    unsigned int num_input;
    unsigned int num_output;
    pearl_version version;
} pearl_network;

PEARL_API pearl_network *pearl_network_create(const unsigned int num_input, const unsigned int num_output);
PEARL_API void pearl_network_destroy(pearl_network **network);
PEARL_API void pearl_network_layer_add(pearl_network **network, const pearl_layer_type type, const int neurons, const pearl_activation_function_type activation_function);
PEARL_API void pearl_network_layer_add_output(pearl_network **network, const pearl_activation_function_type activation_function);
PEARL_API void pearl_network_layer_add_fully_connect(pearl_network **network, const int neurons, const pearl_activation_function_type activation_function);
//PEARL_API void pearl_network_layer_add_dropout(pearl_network **network, const int neurons, pearl_activation_function_type activation_function, const double dropout_rate);
PEARL_API void pearl_network_layers_initialise(pearl_network **network);
PEARL_API double pearl_network_train_epoch(pearl_network **network, const pearl_tensor *input, const pearl_tensor *output);
void pearl_network_forward(pearl_network **network, const pearl_tensor *input, pearl_tensor **z, pearl_tensor **a);
PEARL_API pearl_tensor *pearl_network_calculate(pearl_network **network, const pearl_tensor *input);

#endif // PEARL_NETWORK_H
