#include <pearl_activation.h>

pearl_activation pearl_activation_create(pearl_activation_type type)
{
    pearl_activation activation;
    activation.type = type;
    switch (type) {
        case pearl_activation_type_linear:
            activation.calculate = &pearl_activation_function_linear;
            activation.calculate_derivative = &pearl_activation_function_derivative_linear;
            break;
        case pearl_activation_type_relu:
            activation.calculate = &pearl_activation_function_relu;
            activation.calculate_derivative = &pearl_activation_function_derivative_relu;
            break;
        case pearl_activation_type_tanh:
            activation.calculate = &pearl_activation_function_tanh;
            activation.calculate_derivative = &pearl_activation_function_derivative_tanh;
            break;
        case pearl_activation_type_sigmoid:
            activation.calculate = &pearl_activation_function_sigmoid;
            activation.calculate_derivative = &pearl_activation_function_derivative_sigmoid;
            break;
    }
    return activation;
}

float pearl_activation_function_linear(float input)
{
    return input;
}

float pearl_activation_function_relu(float input)
{
    return input * (input > 0);
}

float pearl_activation_function_tanh(float input)
{
    return tanhf(input);
}

float pearl_activation_function_sigmoid(float input)
{
    return 1.0f / (1.0f + expf(-input));
}

float pearl_activation_function_derivative_linear(float input)
{
    (void)(input); // Suppress warning
    return 1.0f;
}

float pearl_activation_function_derivative_relu(float input)
{
    return input > 0;
}

float pearl_activation_function_derivative_tanh(float input)
{
    return 1.0f - powf(pearl_activation_function_tanh(input), 2.0f);
}

float pearl_activation_function_derivative_sigmoid(float input)
{
    float val = pearl_activation_function_sigmoid(input);
    return val * (1.0f - val);
}
