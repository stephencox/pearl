#ifndef PEARL_ACTIVATION_FUNCTION_H
#define PEARL_ACTIVATION_FUNCTION_H

#include <math.h>
enum pearl_activation_function_type {
    pearl_activation_function_type_linear,
    pearl_activation_function_type_tanh,
    pearl_activation_function_type_sigmoid
};

void *pearl_activation_function_pointer(enum pearl_activation_function_type type);
void *pearl_activation_function_derivative_pointer(enum pearl_activation_function_type type);
double pearl_activation_function_linear(double input);
double pearl_activation_function_tanh(double input);
double pearl_activation_function_sigmoid(double input);
double pearl_activation_function_derivative_linear(double input);
double pearl_activation_function_derivative_tanh(double input);
double pearl_activation_function_derivative_sigmoid(double input);

#endif // PEARL_ACTIVATION_FUNCTION_H
