#include <pearl_activation_function.h>

pearl_activation pearl_activation_create(pearl_activation_function_type type)
{
    pearl_activation activation;
    activation.type = type;
    switch (type) {
        case pearl_activation_function_type_linear:
            activation.calculate = &pearl_activation_function_linear;
            activation.calculate_derivative = &pearl_activation_function_derivative_linear;
            break;
        case pearl_activation_function_type_relu:
            activation.calculate = &pearl_activation_function_relu;
            activation.calculate_derivative = &pearl_activation_function_derivative_relu;
            break;
        case pearl_activation_function_type_tanh:
            activation.calculate = &pearl_activation_function_tanh;
            activation.calculate_derivative = &pearl_activation_function_derivative_tanh;
            break;
        case pearl_activation_function_type_sigmoid:
            activation.calculate = &pearl_activation_function_sigmoid;
            activation.calculate_derivative = &pearl_activation_function_derivative_sigmoid;
            break;
    }
    return activation;
}

double pearl_activation_function_linear(double input)
{
    return input;
}

double pearl_activation_function_relu(double input)
{
    return input * (input > 0);
}

double pearl_activation_function_tanh(double input)
{
    return tanh(input);
}

double pearl_activation_function_sigmoid(double input)
{
    return 1.0 / (1.0 + exp(-input));
}

double pearl_activation_function_derivative_linear(double input)
{
    (void)(input); // Suppress warning
    return 1.0;
}

double pearl_activation_function_derivative_relu(double input)
{
    return input > 0;
}

double pearl_activation_function_derivative_tanh(double input)
{
    return 1 - pow(pearl_activation_function_tanh(input), 2.0);
}

double pearl_activation_function_derivative_sigmoid(double input)
{
    double val = pearl_activation_function_sigmoid(input);
    return val * (1.0 - val);
}
