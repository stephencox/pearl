#include <stdio.h>
//#include <pearl_network.h>
//#include <pearl_tensor.h>
//#include <time.h>
#include <pearl_graph.h>

int main()
{
    /*clock_t t1, t2;
    pearl_network *network = pearl_network_create(2, 1);
    pearl_network_layer_add_fully_connect(&network, 1000, pearl_activation_function_type_relu);
    //pearl_network_layer_add_fully_connect(&network, 1000, pearl_activation_function_type_relu);
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

    double loss;
    t1 = clock();
    for (int i = 0; i < 1000; i++) {
        loss = pearl_network_train_epoch(&network, input, output);
    }
    t2 = clock();
    long elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000.0;
    //pearl_tensor *output_pred = pearl_network_calculate(&network, input);
    //pearl_tensor_reduce_dimension(&output_pred, 1);
    //pearl_tensor_reduce_dimension(&output, 1);
    //double accuracy = pearl_util_accuracy(output, output_pred);
    printf("----------------\nLoss=%f  Accuracy=%3.2f (%ld ms)\n----------------\n", loss, 0.0, elapsed);

    pearl_network_destroy(&network);
    pearl_tensor_destroy(&input);
    pearl_tensor_destroy(&output);
    //pearl_tensor_destroy(&output_pred);*/

    pearl_graph *graph = pearl_graph_create(1);
    pearl_graph_layer *input = pearl_graph_layer_create_input(10);
    pearl_graph_layer *fc1 = pearl_graph_layer_create_fully_connected(10, 10);
    pearl_graph_layer *output = pearl_graph_layer_create_output(10);
    graph->inputs[0] = input;
    pearl_graph_add_child(&input, &fc1);
    pearl_graph_add_child(&fc1, &output);
    pearl_graph_destroy(&graph);

    return 0;
}
