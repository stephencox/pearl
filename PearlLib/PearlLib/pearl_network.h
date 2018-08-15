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
    pearl_layer *input_layer;
    pearl_layer *output_layer;
    pearl_optimiser optimiser;
    pearl_loss loss;
    double learning_rate;
    pearl_version version;
} pearl_network;

PEARL_API pearl_network *pearl_network_create();
PEARL_API void pearl_network_destroy(pearl_network **network);
PEARL_API double pearl_network_train_epoch(pearl_network **network, const pearl_tensor *input, const pearl_tensor *output);
void pearl_network_forward(pearl_network **network, const pearl_tensor *input);
void pearl_network_backward(pearl_network **network, const pearl_tensor *output);
PEARL_API pearl_tensor *pearl_network_calculate(pearl_network **network, const pearl_tensor *input);

#endif // PEARL_NETWORK_H
