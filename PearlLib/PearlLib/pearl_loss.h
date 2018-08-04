#ifndef PEARL_LOSS_H
#define PEARL_LOSS_H

#include <pearl_tensor.h>
#include <math.h>

typedef enum {
    pearl_loss_binary_cross_entropy,
    pearl_loss_mean_squared_error
} pearl_loss;

PEARL_API double pearl_loss_binary_cross_entropy_cost(const pearl_tensor *output, const pearl_tensor *output_prediction);
PEARL_API double pearl_loss_mean_squared_error_cost(const pearl_tensor *output, const pearl_tensor *output_prediction);

#endif // PEARL_LOSS_H
