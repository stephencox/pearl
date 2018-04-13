#include <stdio.h>
#include <pearl_network.h>
#include <pearl_matrix.h>

int main()
{
    pearl_network *network = pearl_network_create();
    pearl_network_layer_add_input(network, 2);
    pearl_network_layer_add_fully_connect(network, 3, pearl_activation_function_type_tanh);
    pearl_network_layer_add_output(network, 1, pearl_activation_function_type_sigmoid);
    pearl_network_layers_initialise(network);

    pearl_matrix *input = pearl_matrix_create(4,2);
    pearl_matrix *output = pearl_matrix_create(4,1);
    int counter_in = 0, counter_out = 0;
    for(int i=0; i<=1; i++){
        for(int j=0; j<=1; j++){
            int a = i & j;
            int b = ~i & ~j;
            input->data[counter_in] = i;
            input->data[counter_in+1] = j;
            output->data[counter_out] = ~a & ~b;
            counter_in+=2;
            counter_out++;
        }
    }

    printf("Input:\n");
    pearl_matrix_print(input);
    printf("Output:\n");
    pearl_matrix_print(output);

    pearl_network_train_epoch(network, input, output);

    pearl_network_destroy(network);
    pearl_matrix_destroy(input);
    pearl_matrix_destroy(output);

    return 0;
}
