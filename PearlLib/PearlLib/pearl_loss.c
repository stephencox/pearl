#include <pearl_loss.h>

PEARL_API double pearl_loss_binary_cross_entropy_cost(const pearl_tensor *output, const pearl_tensor *output_prediction)
{
    assert(output->dimension == 2);
    assert(output->size[0] == 1);
    assert(output_prediction->dimension == 2);
    assert(output_prediction->size[0] == 1);

    double cost = 0.0;
    for (unsigned int i = 0; i < output->size[1]; i++) {
        cost -= output->data[i] * log(output_prediction->data[i]) + (1.0 - output->data[i]) * log(1.0 - output_prediction->data[i]);
    }
    return cost / (double)(output->size[1]);
}

PEARL_API double pearl_loss_mean_squared_error_cost(const pearl_tensor *output, const pearl_tensor *output_prediction)
{
    assert(output->dimension == 2);
    assert(output->size[0] == 1);
    assert(output_prediction->dimension == 2);
    assert(output_prediction->size[0] == 1);

    double cost = 0.0;
    double diff;
    for (unsigned int i = 0; i < output->size[1]; i++) {
        diff = output->data[i] - output_prediction->data[i];
        cost += diff * diff;
    }
    return cost / (double)(output->size[1]);
}
