#ifndef PEARL_LAYER_H
#define PEARL_LAYER_H

#include <stdlib.h>
#include <math.h>
#include <pearl_activation_function.h>
#include <pearl_tensor.h>
#include <pearl_util.h>

#define PEARL_LAYER_VERSION_MAJOR 1
#define PEARL_LAYER_VERSION_MINOR 0
#define PEARL_LAYER_VERSION_REVISION 0

typedef enum {
    pearl_layer_type_input,
    pearl_layer_type_fully_connected,
    pearl_layer_type_dropout
} pearl_layer_type;

typedef struct pearl_layer pearl_layer;
struct pearl_layer {
    pearl_layer_type type;
    void *layer_data;
    unsigned int num_parent_layers;
    pearl_layer **parent_layers;
    unsigned int num_child_layers;
    pearl_layer **child_layers;
    pearl_activation activation;
    unsigned int num_neurons;
    pearl_tensor *z;
    pearl_tensor *a;
    pearl_tensor *dz;
    pearl_tensor *da;
};

typedef struct {
    pearl_tensor *weights;
    pearl_tensor *biases;
    pearl_tensor *dw;
    pearl_tensor *db;
} pearl_layer_data_fully_connected;

typedef struct {
    pearl_tensor *weights;
    double rate;
} pearl_layer_data_dropout;

pearl_layer *pearl_layer_create();
void pearl_layer_destroy(pearl_layer **layer);
PEARL_API void pearl_layer_add_child(pearl_layer **parent, pearl_layer **child);
pearl_layer *pearl_layer_create_input(unsigned int num_neurons);
pearl_layer *pearl_layer_create_fully_connected(unsigned int num_neurons, unsigned int num_neurons_prev_layer);
pearl_layer *pearl_layer_create_dropout(unsigned int num_neurons);
void pearl_layer_forward(pearl_layer **parent_layer, pearl_layer **child_layer);
void pearl_layer_forward_fully_connected(pearl_layer **parent_layer, pearl_layer **child_layer);
void pearl_layer_backward(pearl_layer **child_layer, pearl_layer **parent_layer);
void pearl_layer_backward_fully_connected(pearl_layer **child_layer, pearl_layer **parent_layer);
void pearl_layer_update(pearl_layer *layer, pearl_tensor *dw, pearl_tensor *db, double learning_rate);

#endif // PEARL_LAYER_H
