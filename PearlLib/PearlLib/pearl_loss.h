#ifndef PEARL_LOSS_H
#define PEARL_LOSS_H

#include <pearl_tensor.h>
#include <math.h>

typedef enum {
    pearl_loss_binary_cross_entropy,
    pearl_loss_mean_squared_error
} pearl_loss_type;

typedef struct {
    pearl_loss_type type;
    double (*calculate)(double, double);
    double (*calculate_derivative)(double, double);
} pearl_loss;

pearl_loss pearl_loss_create(pearl_loss_type type);
double pearl_loss_cost(pearl_loss loss, const pearl_tensor *output, const pearl_tensor *output_prediction);
double pearl_loss_binary_cross_entropy_func(double out, double pred);
double pearl_loss_mean_squared_error_func(double out, double pred);
double pearl_loss_binary_cross_entropy_func_derivative(double out, double pred);
double pearl_loss_mean_squared_error_func_derivative(double out, double pred);

#endif // PEARL_LOSS_H
