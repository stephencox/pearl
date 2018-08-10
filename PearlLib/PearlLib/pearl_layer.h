#ifndef PEARL_LAYER_H
#define PEARL_LAYER_H

#include <stdlib.h>
#include <math.h>
#include <pearl_activation_function.h>
#include <pearl_tensor.h>
#include <pearl_version.h>
#include <pearl_util.h>

#define PEARL_LAYER_VERSION_MAJOR 1
#define PEARL_LAYER_VERSION_MINOR 0
#define PEARL_LAYER_VERSION_REVISION 0

typedef enum pearl_layer_type {
    pearl_layer_type_fully_connect,
    //pearl_layer_type_dropout
} pearl_layer_type;

typedef struct {
    pearl_version version;
    pearl_layer_type type;
    pearl_activation_function_type activation_function;
    unsigned int num_neurons;
    //double dropout_rate;
    pearl_tensor *weights;
    pearl_tensor *biases;
} pearl_layer;

void pearl_layer_initialise(pearl_layer **layer, const int num_neurons_prev_layer);
void pearl_layer_destroy(pearl_layer **layer);
void pearl_layer_forward(pearl_layer **layer, const pearl_tensor *input, pearl_tensor **z, pearl_tensor **a);
void pearl_layer_backward(const pearl_layer *layer, const pearl_tensor *dz, const pearl_tensor *a, pearl_tensor **dw, pearl_tensor **db, pearl_tensor **da_prev);
void pearl_layer_update(pearl_layer *layer, pearl_tensor *dw, pearl_tensor *db, double learning_rate);

#endif // PEARL_LAYER_H
