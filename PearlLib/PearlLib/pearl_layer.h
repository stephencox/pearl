#ifndef PEARL_LAYER_H
#define PEARL_LAYER_H

#include <stdlib.h>
#include <math.h>
#include <pearl_activation_function.h>
#include <pearl_matrix.h>
#include <pearl_vector.h>

enum pearl_layer_type {
    pearl_layer_type_input,
    pearl_layer_type_fully_connect,
    pearl_layer_type_dropout,
    pearl_layer_type_output
};

struct pearl_layer {
    enum pearl_layer_type type;
    enum pearl_activation_function_type activation_function;
    int neurons;
    double dropout_rate;
    pearl_vector *weights;
    pearl_vector *biases;
};

void pearl_layer_initialise(struct pearl_layer *layer, const struct pearl_layer *prev_layer);
void pearl_layer_destroy(struct pearl_layer *layer);
struct pearl_matrix *pearl_layer_forward(struct pearl_layer *layer, const struct pearl_matrix *input);

#endif // PEARL_LAYER_H
