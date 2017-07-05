#include <stdio.h>
#include <pearl_network.h>

int main()
{
    struct pearl_network *network = pearl_network_create();
    pearl_network_layer_add_input(network, 2);
    pearl_network_layer_add_dropout(network, 4, pearl_activation_function_type_tanh, 0.1);
    pearl_network_layer_add_output(network, 1, pearl_activation_function_type_linear);
    pearl_network_destroy(network);
    return 0;
}
