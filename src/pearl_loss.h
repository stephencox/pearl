#pragma once

#include <pearl_tensor.h>
#include <math.h>

typedef enum {
    pearl_loss_binary_cross_entropy,
    pearl_loss_mean_squared_error
} pearl_loss_type;

typedef struct {
    pearl_loss_type type;
    float (*calculate)(float, float);
    float (*calculate_derivative)(float, float);
} pearl_loss;

PEARL_API pearl_loss pearl_loss_create(pearl_loss_type type);
float pearl_loss_cost(pearl_loss loss, const pearl_tensor *output, const pearl_tensor *output_prediction);
float pearl_loss_binary_cross_entropy_func(float out, float pred);
float pearl_loss_mean_squared_error_func(float out, float pred);
float pearl_loss_binary_cross_entropy_func_derivative(float out, float pred);
float pearl_loss_mean_squared_error_func_derivative(float out, float pred);
