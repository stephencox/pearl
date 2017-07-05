#ifndef PEARL_LAYER_H
#define PEARL_LAYER_H

#include <pearl_activation_function.h>

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
};

#endif // PEARL_LAYER_H
