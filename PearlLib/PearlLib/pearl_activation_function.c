#include <pearl_activation_function.h>

void *pearl_activation_function_pointer(pearl_activation_function_type type)
{
    double (*activationFunctionPtr)(double);
    switch (type) {
        case pearl_activation_function_type_linear:
            activationFunctionPtr = &pearl_activation_function_linear;
            break;
        case pearl_activation_function_type_relu:
            activationFunctionPtr = &pearl_activation_function_relu;
            break;
        case pearl_activation_function_type_tanh:
            activationFunctionPtr = &pearl_activation_function_tanh;
            break;
        case pearl_activation_function_type_sigmoid:
            activationFunctionPtr = &pearl_activation_function_sigmoid;
            break;
        default:
            activationFunctionPtr = &pearl_activation_function_tanh;
            break;
    }
    return activationFunctionPtr;
}

void *pearl_activation_function_derivative_pointer(pearl_activation_function_type type)
{
    double (*activationFunctionPtr)(double);
    switch (type) {
        case pearl_activation_function_type_linear:
            activationFunctionPtr = &pearl_activation_function_derivative_linear;
            break;
        case pearl_activation_function_type_relu:
            activationFunctionPtr = &pearl_activation_function_derivative_relu;
            break;
        case pearl_activation_function_type_tanh:
            activationFunctionPtr = &pearl_activation_function_derivative_tanh;
            break;
        case pearl_activation_function_type_sigmoid:
            activationFunctionPtr = &pearl_activation_function_derivative_sigmoid;
            break;
        default:
            activationFunctionPtr = &pearl_activation_function_derivative_tanh;
            break;
    }
    return activationFunctionPtr;
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
    return pearl_activation_function_sigmoid(input) * pearl_activation_function_sigmoid(1.0 - input);
}
