#ifndef PEARL_GRAPH_H
#define PEARL_GRAPH_H

#include <pearl_activation_function.h>
#include <pearl_tensor.h>
#include <pearl_util.h>

typedef enum {
    pearl_graph_layer_type_input,
    pearl_graph_layer_type_fully_connected,
    pearl_graph_layer_type_dropout,
    pearl_graph_layer_type_output
} pearl_graph_layer_type;

typedef struct pearl_graph_layer {
    pearl_graph_layer_type node_type;
    void *node_data;
    unsigned int num_parents;
    struct pearl_graph_layer **parents;
    unsigned int num_childs;
    struct pearl_graph_layer **childs;
} pearl_graph_layer; //TODO: merge with pearl_layer

typedef struct {
    unsigned int num_layers;
    unsigned int num_inputs;
    pearl_graph_layer **inputs;
} pearl_graph; //TODO: Merge with pearl_network

typedef struct {
    pearl_activation_function_type activation_function;
    unsigned int num_neurons;
} pearl_graph_node_input;

typedef struct {
    pearl_activation_function_type activation_function;
    unsigned int num_neurons;
} pearl_graph_node_output;

typedef struct {
    pearl_activation_function_type activation_function;
    unsigned int num_neurons;
    pearl_tensor *weights;
    pearl_tensor *biases;
} pearl_graph_node_fully_connected;

typedef struct {
    unsigned int num_neurons;
    pearl_tensor *weights;
} pearl_graph_node_dropout;

PEARL_API pearl_graph *pearl_graph_create(unsigned int num_inputs);
PEARL_API void pearl_graph_destroy(pearl_graph **graph);
PEARL_API void pearl_graph_add_child(pearl_graph_layer **parent, pearl_graph_layer **child);

pearl_graph_layer *pearl_graph_layer_create();
void pearl_graph_layer_destroy(pearl_graph_layer **layer);

pearl_graph_layer *pearl_graph_layer_create_input(unsigned int num_neurons);
pearl_graph_layer *pearl_graph_layer_create_output(unsigned int num_neurons);
pearl_graph_layer *pearl_graph_layer_create_fully_connected(unsigned int num_neurons, unsigned int num_neurons_prev_layer);
pearl_graph_layer *pearl_graph_layer_create_dropout(unsigned int num_neurons);

#endif // PEARL_GRAPH_H
