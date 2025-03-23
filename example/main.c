#include <stdio.h>
#include <pearl_network.h>
#include <time.h>

int main()
{
    pearl_network *network = pearl_network_create();
    pearl_layer *input_layer = pearl_layer_create_input(2);
    network->input_layer = input_layer;
    pearl_layer *fc1 = pearl_layer_create_fully_connected(100, 2);
    pearl_layer_add_child(&input_layer, &fc1);
    pearl_layer *fc2 = pearl_layer_create_fully_connected(100, 100);
    pearl_layer_add_child(&fc1, &fc2);
    pearl_layer *output_layer = pearl_layer_create_fully_connected(1, 100);
    output_layer->activation = pearl_activation_create(pearl_activation_type_sigmoid);
    pearl_layer_add_child(&fc2, &output_layer);
    network->output_layer = output_layer;
    network->learning_rate = 0.1f;

    pearl_tensor *input = pearl_tensor_create(2, 4, 2);
    pearl_tensor *output = pearl_tensor_create(2, 1, 4);
    int counter_in = 0;
    int counter_out = 0;
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

    float loss;
    clock_t t1, t2;
    t1 = clock();
    for (int i = 0; i < 1000; i++) {
        loss = pearl_network_train_epoch(&network, input, output);
    }
    t2 = clock();
    printf("----------------\nLoss=%f (%ld ms)\n----------------\n", loss, (long)(((float)t2 - t1) / CLOCKS_PER_SEC * 1000));

    return 0;
}
