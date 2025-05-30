#pragma once

#include <pearl_global.h>
#include <math.h>

typedef enum {
    pearl_activation_type_linear,
    pearl_activation_type_relu,
    pearl_activation_type_tanh,
    pearl_activation_type_sigmoid
} pearl_activation_type;

typedef struct {
    pearl_activation_type type;
    float (*calculate)(float);
    float (*calculate_derivative)(float);
} pearl_activation;

PEARL_API pearl_activation pearl_activation_create(pearl_activation_type type);
float pearl_activation_function_linear(float input);
float pearl_activation_function_relu(float input);
float pearl_activation_function_tanh(float input);
float pearl_activation_function_sigmoid(float input);
float pearl_activation_function_derivative_linear(float input);
float pearl_activation_function_derivative_relu(float input);
float pearl_activation_function_derivative_tanh(float input);
float pearl_activation_function_derivative_sigmoid(float input);
