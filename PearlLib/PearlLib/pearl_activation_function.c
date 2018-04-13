#include <pearl_activation_function.h>

void *pearl_activation_function_pointer(enum pearl_activation_function_type type)
{
    double (*activationFunctionPtr)(double);
    switch (type) {
        case pearl_activation_function_type_tanh:
            activationFunctionPtr = &pearl_activation_function_tanh;
            break;
        case pearl_activation_function_type_sigmoid:
            activationFunctionPtr = &pearl_activation_function_sigmoid;
            break;
        case pearl_activation_function_type_linear:
            activationFunctionPtr = &pearl_activation_function_tanh;
            break;
        default:
            activationFunctionPtr = &pearl_activation_function_tanh;
            break;
    }
    return activationFunctionPtr;
}

double pearl_activation_function_tanh(double input)
{
    return tanh(input);
}

double pearl_activation_function_sigmoid(double input)
{
    return 1.0 / (1.0 + exp(-input));
}
