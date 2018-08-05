#include <pearl_loss.h>

pearl_loss pearl_loss_create(pearl_loss_type type)
{
    pearl_loss loss;
    loss.type = type;
    switch (type) {
        case pearl_loss_binary_cross_entropy:
            loss.calculate = &pearl_loss_binary_cross_entropy_func;
            loss.calculate_derivative = &pearl_loss_binary_cross_entropy_func_derivative;
            break;
        case pearl_loss_mean_squared_error:
            loss.calculate = &pearl_loss_mean_squared_error_func;
            loss.calculate_derivative = &pearl_loss_mean_squared_error_func_derivative;
            break;
    }
    return loss;
}

double pearl_loss_cost(pearl_loss loss, const pearl_tensor *output, const pearl_tensor *output_prediction)
{
    assert(output->dimension == 2);
    assert(output->size[0] == 1);
    assert(output_prediction->dimension == 2);
    assert(output_prediction->size[0] == 1);

    double cost = 0.0;
    for (unsigned int i = 0; i < output->size[1]; i++) {
        cost += loss.calculate(output->data[i], output_prediction->data[i]);
    }
    return cost / (double)(output->size[1]);
}

double pearl_loss_binary_cross_entropy_func(double out, double pred)
{
    return -(out * log(pred) + (1.0 - out) * log(1.0 - pred));
}

double pearl_loss_binary_cross_entropy_func_derivative(double out, double pred)
{
    return  out / pred - (1.0 - out) / (1.0 - pred);
}

double pearl_loss_mean_squared_error_func(double out, double pred)
{
    return pow(out - pred, 2);
}

double pearl_loss_mean_squared_error_func_derivative(double out, double pred)
{
    return 0.5 * (out - pred);
}
