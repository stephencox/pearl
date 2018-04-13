#ifndef PEARL_LAYER_H
#define PEARL_LAYER_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <pearl_activation_function.h>
#include <pearl_matrix.h>
#include <pearl_vector.h>
#include <pearl_activation_function.h>

enum pearl_layer_type {
    pearl_layer_type_input,
    pearl_layer_type_fully_connect,
    pearl_layer_type_dropout,
    pearl_layer_type_output
};

typedef struct {
    enum pearl_layer_type type;
    enum pearl_activation_function_type activation_function;
    int neurons;
    double dropout_rate;
    pearl_matrix *weights;
    pearl_vector *biases;
} pearl_layer;

void pearl_layer_initialise(pearl_layer *layer, const pearl_layer *prev_layer);
void pearl_layer_destroy(pearl_layer *layer);
void pearl_layer_print(pearl_layer *layer);
void pearl_layer_forward(pearl_layer *layer, const pearl_matrix *input, pearl_matrix *a, pearl_matrix *z);

#endif // PEARL_LAYER_H
