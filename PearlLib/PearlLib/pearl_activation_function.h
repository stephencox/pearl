#ifndef PEARL_ACTIVATION_FUNCTION_H
#define PEARL_ACTIVATION_FUNCTION_H

#include <pearl_global.h>
#include <math.h>

typedef enum {
    pearl_activation_function_type_linear,
    pearl_activation_function_type_relu,
    pearl_activation_function_type_tanh,
    pearl_activation_function_type_sigmoid
} pearl_activation_function_type;

typedef struct {
    pearl_activation_function_type type;
    double (*calculate)(double);
    double (*calculate_derivative)(double);
} pearl_activation;

PEARL_API pearl_activation pearl_activation_create(pearl_activation_function_type type);
double pearl_activation_function_linear(double input);
double pearl_activation_function_relu(double input);
double pearl_activation_function_tanh(double input);
double pearl_activation_function_sigmoid(double input);
double pearl_activation_function_derivative_linear(double input);
double pearl_activation_function_derivative_relu(double input);
double pearl_activation_function_derivative_tanh(double input);
double pearl_activation_function_derivative_sigmoid(double input);

#endif // PEARL_ACTIVATION_FUNCTION_H
