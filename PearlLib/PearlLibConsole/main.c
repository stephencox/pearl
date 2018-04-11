#include <stdio.h>
#include <pearl_network.h>

#include <pearl_matrix.h>
int main()
{
    struct pearl_network *network = pearl_network_create();
    pearl_network_layer_add_input(network, 2);
    pearl_network_layer_add_fully_connect(network, 4, pearl_activation_function_type_tanh);
    pearl_network_layer_add_output(network, 1, pearl_activation_function_type_linear);
    pearl_network_layers_initialise(network);

    struct pearl_matrix *input = pearl_matrix_create(2,4);
    struct pearl_matrix *output = pearl_matrix_create(1,4);
    int counter = 0;
    for(int i=0; i<=1; i++){
        for(int j=0; j<=1; j++){
            int a = i & j;
            int b = ~i & ~j;
            input->data[counter] = i;
            input->data[counter+1] = j;
            output->data[counter] = ~a & ~b;
        }
    }

    pearl_network_train_epoch(network, input, output);

    pearl_network_destroy(network);
    pearl_matrix_destroy(input);
    pearl_matrix_destroy(output);

    return 0;
}
