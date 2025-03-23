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

float pearl_loss_cost(pearl_loss loss, const pearl_tensor *output, const pearl_tensor *output_prediction)
{
    assert(output->dimension == 2);
    assert(output->size[0] == 1);
    assert(output_prediction->dimension == 2);
    assert(output_prediction->size[0] == 1);

    float cost = 0.0f;
    for (unsigned int i = 0; i < output->size[1]; i++) {
        cost += loss.calculate(output->data[i], output_prediction->data[i]);
    }
    return cost / (float)(output->size[1]);
}

float pearl_loss_binary_cross_entropy_func(float out, float pred)
{
    return -(out * logf(pred) + (1.0f - out) * logf(1.0f - pred));
}

float pearl_loss_binary_cross_entropy_func_derivative(float out, float pred)
{
    return  out / pred - (1.0f - out) / (1.0f - pred);
}

float pearl_loss_mean_squared_error_func(float out, float pred)
{
    return powf(out - pred, 2.0f);
}

float pearl_loss_mean_squared_error_func_derivative(float out, float pred)
{
    return 0.5f * (out - pred);
}
