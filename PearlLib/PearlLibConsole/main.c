#include <stdio.h>
#include <pearl_network.h>
#include <pearl_tensor.h>

int main()
{
    pearl_network *network = pearl_network_create(2, 1);
    pearl_network_layer_add_fully_connect(&network, 3, pearl_activation_function_type_relu);
    pearl_network_layer_add_output(&network, pearl_activation_function_type_sigmoid);
    pearl_network_layers_initialise(&network);
    network->learning_rate = 0.1;

    pearl_tensor *input = pearl_tensor_create(2, 4, 2);
    pearl_tensor *output = pearl_tensor_create(2, 1, 4);
    int counter_in = 0, counter_out = 0;
    for (int i = 0; i <= 1; i++) {
        for (int j = 0; j <= 1; j++) {
            int a = i & j;
            int b = ~i & ~j;
            input->data[counter_in] = i;
            input->data[counter_in + 1] = j;
            output->data[counter_out] = ~a & ~b;
            counter_in += 2;
            counter_out++;
        }
    }

    network->layers[0]->biases->data[0] = 0.11;
    network->layers[0]->biases->data[1] = 0.12;
    network->layers[0]->biases->data[2] = 0.13;
    network->layers[1]->biases->data[0] = 0.14;

    network->layers[0]->weights->data[0] = 0.21;
    network->layers[0]->weights->data[1] = 0.22;
    network->layers[0]->weights->data[2] = 0.23;
    network->layers[0]->weights->data[3] = 0.24;
    network->layers[0]->weights->data[4] = 0.25;
    network->layers[0]->weights->data[5] = 0.26;
    network->layers[1]->weights->data[0] = 0.15;
    network->layers[1]->weights->data[1] = 0.16;
    network->layers[1]->weights->data[2] = 0.17;

    for (int i = 0; i < 1000; i++) {
        pearl_network_train_epoch(&network, input, output);
    }

    pearl_network_destroy(&network);
    pearl_tensor_destroy(&input);
    pearl_tensor_destroy(&output);

    return 0;
}
