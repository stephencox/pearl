#include <pearl_loss.h>

double pearl_loss_binary_cross_entropy_cost(const pearl_tensor *output, const pearl_tensor *output_prediction)
{
    assert(output->dimension == 2);
    assert(output->size[1] == 1);
    assert(output_prediction->dimension == 2);
    assert(output_prediction->size[0] == 1);

    double cost = 0.0;
    for (unsigned int i = 0; i < output->size[0]; i++) {
        if (output->data[i] > 0.0) {
            cost += log(output_prediction->data[i]);
        }
        else {
            cost += log(1.0 - output_prediction->data[i]);
        }
    }
    return -cost / (double)(output->size[1]);
}
