#include <pearl_activation_function.h>

void *pearl_activation_function_pointer(enum pearl_activation_function_type type){
    double (*activationFunctionPtr)(double);
    switch (type) {
        case pearl_activation_function_type_tanh:
            activationFunctionPtr = &pearl_activation_function_tanh;
            break;
        case pearl_activation_function_type_sigmoid:
            activationFunctionPtr = &tanh;
            break;
        case pearl_activation_function_type_linear:
            activationFunctionPtr = &tanh;
            break;
        default:
            activationFunctionPtr = &tanh;
            break;
    }
    return activationFunctionPtr;
}

double pearl_activation_function_tanh(double input){
    return tanh(input);
}
