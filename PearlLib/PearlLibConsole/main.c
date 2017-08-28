#include <stdio.h>
#include <pearl_network.h>

int main()
{
    struct pearl_network *network = pearl_network_create();
    pearl_network_layer_add_input(network, 20000);
    pearl_network_layer_add_fully_connect(network, 42000, pearl_activation_function_type_tanh);
    pearl_network_layer_add_output(network, 1, pearl_activation_function_type_linear);
    pearl_network_layers_initialise(network);
    pearl_network_destroy(network);
    return 0;
}
