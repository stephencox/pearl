#include "unity.h"
#include <pearl_network.h>
#include <pearl_json.h>
#include <pearl_print.h>

void setUp(void)
{
}

void tearDown(void)
{
}

void test_network_create()
{
    pearl_network *network = pearl_network_create();
    TEST_ASSERT_NOT_NULL(network);
    TEST_ASSERT_NULL(network->input_layer);
    TEST_ASSERT_EQUAL_FLOAT(network->learning_rate, 1e-3);
    TEST_ASSERT_EQUAL_INT(network->loss.type, pearl_loss_binary_cross_entropy);
    TEST_ASSERT_EQUAL_INT(network->optimiser, pearl_optimiser_sgd);
    TEST_ASSERT_EQUAL_UINT(network->version.major, PEARL_NETWORK_VERSION_MAJOR);
    TEST_ASSERT_EQUAL_UINT(network->version.minor, PEARL_NETWORK_VERSION_MINOR);
    TEST_ASSERT_EQUAL_UINT(network->version.revision, PEARL_NETWORK_VERSION_REVISION);
    pearl_network_destroy(&network);
    TEST_ASSERT_NULL(network);
}

void test_network_add_layers()
{
    pearl_network *network = pearl_network_create();
    pearl_layer *input = pearl_layer_create_input(5);
    network->input_layer = input;
    TEST_ASSERT_EQUAL_INT(input->type, pearl_layer_type_input);
    TEST_ASSERT_EQUAL_INT(input->num_child_layers, 0);
    TEST_ASSERT_NULL(input->child_layers);
    TEST_ASSERT_EQUAL_INT(input->num_parent_layers, 0);
    TEST_ASSERT_NULL(input->parent_layers);
    TEST_ASSERT_NULL(input->layer_data);
    TEST_ASSERT_EQUAL_INT(input->num_neurons, 5);
    TEST_ASSERT_EQUAL_INT(input->activation.type, pearl_activation_type_linear);

    pearl_layer *drop = pearl_layer_create_dropout(5);
    pearl_layer_add_child(&input, &drop);
    TEST_ASSERT_EQUAL_INT(input->num_parent_layers, 0);
    TEST_ASSERT_NULL(input->parent_layers);
    TEST_ASSERT_EQUAL_INT(input->num_child_layers, 1);
    TEST_ASSERT_NOT_NULL(input->child_layers);
    TEST_ASSERT_NOT_NULL(input->child_layers[0]);
    TEST_ASSERT_EQUAL_PTR(input->child_layers[0], drop);
    TEST_ASSERT_EQUAL_INT(drop->type, pearl_layer_type_dropout);
    TEST_ASSERT_EQUAL_INT(drop->num_child_layers, 0);
    TEST_ASSERT_NULL(drop->child_layers);
    TEST_ASSERT_EQUAL_INT(drop->num_parent_layers, 1);
    TEST_ASSERT_NOT_NULL(drop->parent_layers);
    TEST_ASSERT_NOT_NULL(drop->parent_layers[0]);
    TEST_ASSERT_EQUAL_INT(drop->parent_layers[0]->type, pearl_layer_type_input);
    TEST_ASSERT_NOT_NULL(drop->layer_data);
    const pearl_layer_data_dropout *data_drop = (pearl_layer_data_dropout *)drop->layer_data;
    TEST_ASSERT_EQUAL_INT(drop->num_neurons, 5);
    TEST_ASSERT_EQUAL_FLOAT(data_drop->rate, 0.5);
    TEST_ASSERT_EQUAL_INT(data_drop->weights->dimension, 1);
    TEST_ASSERT_NOT_NULL(data_drop->weights->size);
    TEST_ASSERT_EQUAL_INT(data_drop->weights->size[0], 5);
    TEST_ASSERT_NOT_NULL(data_drop->weights->data);
    for (unsigned int i = 0; i < data_drop->weights->size[0]; i++) {
        TEST_ASSERT_EQUAL_FLOAT(data_drop->weights->data[i], 0.0);
    }

    pearl_layer *fc = pearl_layer_create_fully_connected(5, 1);
    pearl_layer_add_child(&drop, &fc);
    TEST_ASSERT_EQUAL_INT(drop->num_child_layers, 1);
    TEST_ASSERT_NOT_NULL(drop->child_layers);
    TEST_ASSERT_NOT_NULL(drop->child_layers[0]);
    TEST_ASSERT_EQUAL_PTR(drop->child_layers[0], fc);
    TEST_ASSERT_EQUAL_INT(fc->type, pearl_layer_type_fully_connected);
    TEST_ASSERT_EQUAL_INT(fc->num_child_layers, 0);
    TEST_ASSERT_NULL(fc->child_layers);
    TEST_ASSERT_EQUAL_INT(fc->num_parent_layers, 1);
    TEST_ASSERT_NOT_NULL(fc->parent_layers);
    TEST_ASSERT_NOT_NULL(fc->parent_layers[0]);
    TEST_ASSERT_EQUAL_INT(fc->parent_layers[0]->type, pearl_layer_type_dropout);
    TEST_ASSERT_NOT_NULL(fc->layer_data);
    const pearl_layer_data_fully_connected *data_fc = (pearl_layer_data_fully_connected *)fc->layer_data;
    TEST_ASSERT_EQUAL_INT(fc->num_neurons, 5);
    TEST_ASSERT_EQUAL_INT(fc->activation.type, pearl_activation_type_relu);
    TEST_ASSERT_NOT_NULL(data_fc->weights);
    TEST_ASSERT_EQUAL_INT(data_fc->weights->dimension, 2);
    TEST_ASSERT_NOT_NULL(data_fc->weights->size);
    TEST_ASSERT_EQUAL_INT(data_fc->weights->size[0], 5);
    TEST_ASSERT_EQUAL_INT(data_fc->weights->size[1], 1);
    TEST_ASSERT_NOT_NULL(data_fc->weights->data);
    for (unsigned int i = 0; i < data_fc->weights->size[0]; i++) {
        TEST_ASSERT_NOT_EQUAL_FLOAT(data_fc->weights->data[i], 0.0);
    }
    TEST_ASSERT_NOT_NULL(data_fc->biases);
    TEST_ASSERT_EQUAL_INT(data_fc->biases->dimension, 1);
    TEST_ASSERT_NOT_NULL(data_fc->biases->size);
    TEST_ASSERT_EQUAL_INT(data_fc->biases->size[0], 5);
    TEST_ASSERT_NOT_NULL(data_fc->biases->data);
    for (unsigned int i = 0; i < data_fc->biases->size[0]; i++) {
        TEST_ASSERT_EQUAL_FLOAT(data_fc->biases->data[i], 0.0);
    }

    pearl_layer *output = pearl_layer_create_fully_connected(1, 5);
    output->activation = pearl_activation_create(pearl_activation_type_linear);
    pearl_layer_add_child(&fc, &output);
    TEST_ASSERT_EQUAL_INT(fc->num_child_layers, 1);
    TEST_ASSERT_NOT_NULL(fc->child_layers);
    TEST_ASSERT_NOT_NULL(fc->child_layers[0]);
    TEST_ASSERT_EQUAL_PTR(fc->child_layers[0], output);
    TEST_ASSERT_EQUAL_INT(output->type, pearl_layer_type_fully_connected);
    TEST_ASSERT_EQUAL_INT(output->num_child_layers, 0);
    TEST_ASSERT_NULL(output->child_layers);
    TEST_ASSERT_EQUAL_INT(output->num_parent_layers, 1);
    TEST_ASSERT_NOT_NULL(output->parent_layers);
    TEST_ASSERT_NOT_NULL(output->parent_layers[0]);
    TEST_ASSERT_EQUAL_INT(output->parent_layers[0]->type, pearl_layer_type_fully_connected);
    TEST_ASSERT_NOT_NULL(output->layer_data);
    TEST_ASSERT_NOT_NULL(output->layer_data);
    TEST_ASSERT_EQUAL_INT(output->num_neurons, 1);
    TEST_ASSERT_EQUAL_INT(output->activation.type, pearl_activation_type_linear);

    pearl_network_destroy(&network);
    TEST_ASSERT_NULL(network);
}

void test_network_save_load()
{
    pearl_network *network = pearl_network_create();
    pearl_layer *input = pearl_layer_create_input(5);
    network->input_layer = input;
    pearl_layer *drop = pearl_layer_create_dropout(5);
    pearl_layer_add_child(&input, &drop);
    pearl_layer *fc = pearl_layer_create_fully_connected(5, 1);
    pearl_layer_add_child(&drop, &fc);
    pearl_json_network_serialise("network.json", network);
    pearl_network_destroy(&network);

    pearl_network *network_load = pearl_json_network_deserialise("network.json");
    TEST_ASSERT_NOT_NULL(network_load);

    TEST_ASSERT_NOT_NULL(network_load->input_layer);
    const pearl_layer *input_load = network_load->input_layer;
    TEST_ASSERT_EQUAL_INT(input_load->type, pearl_layer_type_input);
    TEST_ASSERT_EQUAL_INT(input_load->num_parent_layers, 0);
    TEST_ASSERT_NULL(input_load->parent_layers);
    TEST_ASSERT_EQUAL_INT(input_load->num_child_layers, 1);
    TEST_ASSERT_NULL(input_load->parent_layers);
    TEST_ASSERT_NULL(input_load->layer_data);
    TEST_ASSERT_EQUAL_INT(input_load->num_neurons, 5);
    TEST_ASSERT_EQUAL_INT(input_load->activation.type, pearl_activation_type_linear);

    TEST_ASSERT_NOT_NULL(input_load->child_layers);
    TEST_ASSERT_NOT_NULL(input_load->child_layers[0]);
    pearl_layer *drop_load = input_load->child_layers[0];
    TEST_ASSERT_EQUAL_INT(drop_load->type, pearl_layer_type_dropout);
    TEST_ASSERT_EQUAL_INT(drop_load->num_child_layers, 1);
    TEST_ASSERT_NOT_NULL(drop_load->child_layers);
    TEST_ASSERT_NOT_NULL(drop_load->child_layers[0]);
    TEST_ASSERT_EQUAL_INT(drop_load->num_parent_layers, 1);
    TEST_ASSERT_NOT_NULL(drop_load->parent_layers);
    TEST_ASSERT_NOT_NULL(drop_load->parent_layers[0]);
    TEST_ASSERT_EQUAL_INT(drop_load->parent_layers[0]->type, pearl_layer_type_input);
    TEST_ASSERT_NOT_NULL(drop_load->layer_data);
    TEST_ASSERT_EQUAL_INT(drop_load->num_neurons, 5);
    const pearl_layer_data_dropout *data_drop_load = (pearl_layer_data_dropout *)drop_load->layer_data;
    TEST_ASSERT_EQUAL_FLOAT(data_drop_load->rate, 0.5);
    TEST_ASSERT_EQUAL_INT(data_drop_load->weights->dimension, 1);
    TEST_ASSERT_NOT_NULL(data_drop_load->weights->size);
    TEST_ASSERT_EQUAL_INT(data_drop_load->weights->size[0], 5);
    TEST_ASSERT_NOT_NULL(data_drop_load->weights->data);
    for (unsigned int i = 0; i < data_drop_load->weights->size[0]; i++) {
        TEST_ASSERT_EQUAL_FLOAT(data_drop_load->weights->data[i], 0.0);
    }

    //TODO: Compare biases and weights
    TEST_ASSERT_NOT_NULL(drop_load->child_layers);
    TEST_ASSERT_NOT_NULL(drop_load->child_layers[0]);
    pearl_layer *fc_load = drop_load->child_layers[0];
    TEST_ASSERT_EQUAL_INT(fc_load->type, pearl_layer_type_fully_connected);
    TEST_ASSERT_EQUAL_INT(fc_load->num_child_layers, 0);
    TEST_ASSERT_NULL(fc_load->child_layers);
    TEST_ASSERT_EQUAL_INT(fc_load->num_parent_layers, 1);
    TEST_ASSERT_NOT_NULL(fc_load->parent_layers);
    TEST_ASSERT_NOT_NULL(fc_load->parent_layers[0]);
    TEST_ASSERT_EQUAL_INT(fc_load->parent_layers[0]->type, pearl_layer_type_dropout);
    TEST_ASSERT_EQUAL_INT(fc_load->num_neurons, 5);
    TEST_ASSERT_EQUAL_INT(fc_load->activation.type, pearl_activation_type_relu);
    TEST_ASSERT_NOT_NULL(fc_load->layer_data);
    const pearl_layer_data_fully_connected *data_fc_load = (pearl_layer_data_fully_connected *)fc_load->layer_data;
    TEST_ASSERT_NOT_NULL(data_fc_load->weights);
    TEST_ASSERT_EQUAL_INT(data_fc_load->weights->dimension, 2);
    TEST_ASSERT_NOT_NULL(data_fc_load->weights->size);
    TEST_ASSERT_EQUAL_INT(data_fc_load->weights->size[0], 5);
    TEST_ASSERT_EQUAL_INT(data_fc_load->weights->size[1], 1);
    TEST_ASSERT_NOT_NULL(data_fc_load->weights->data);
    for (unsigned int i = 0; i < data_fc_load->weights->size[0]; i++) {
        TEST_ASSERT_NOT_EQUAL_FLOAT(data_fc_load->weights->data[i], 0.0f);
    }
    TEST_ASSERT_NOT_NULL(data_fc_load->biases);
    TEST_ASSERT_EQUAL_INT(data_fc_load->biases->dimension, 1);
    TEST_ASSERT_NOT_NULL(data_fc_load->biases->size);
    TEST_ASSERT_EQUAL_INT(data_fc_load->biases->size[0], 5);
    TEST_ASSERT_NOT_NULL(data_fc_load->biases->data);
    for (unsigned int i = 0; i < data_fc_load->biases->size[0]; i++) {
        TEST_ASSERT_EQUAL_FLOAT(data_fc_load->biases->data[i], 0.0f);
    }

    pearl_network_destroy(&network_load);
}

void test_network_epoch_check()
{
    pearl_network *network = pearl_network_create();
    pearl_layer *input_layer = pearl_layer_create_input(2);
    network->input_layer = input_layer;
    pearl_layer *fc = pearl_layer_create_fully_connected(3, 2);
    pearl_layer_add_child(&input_layer, &fc);
    pearl_layer *output_layer = pearl_layer_create_fully_connected(1, 3);
    output_layer->activation = pearl_activation_create(pearl_activation_type_sigmoid);
    pearl_layer_add_child(&fc, &output_layer);
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
            input->data[counter_in] = (float)i;
            input->data[counter_in + 1] = (float)j;
            output->data[counter_out] = (float)(~a & ~b);
            counter_in += 2;
            counter_out++;
        }
    }
    pearl_layer_data_fully_connected *fc_data = (pearl_layer_data_fully_connected *)fc->layer_data;
    pearl_layer_data_fully_connected *output_data = (pearl_layer_data_fully_connected *)output_layer->layer_data;
    fc_data->weights->data[0] = -0.2670205687292891f;
    fc_data->weights->data[1] = 0.4848471037990288f;
    fc_data->weights->data[2] = -0.6315295129156053f;
    fc_data->weights->data[3] = -0.7420722005600799f;
    fc_data->weights->data[4] = 1.3681090910152733f;
    fc_data->weights->data[5] = -1.232025382027168f;
    output_data->weights->data[0] = 0.0733915213686632f;
    output_data->weights->data[1] = -1.004698165338342f;
    output_data->weights->data[2] = -0.1679168826901067f;

    float loss = pearl_network_train_epoch(&network, input, output);
    TEST_ASSERT_EQUAL_FLOAT(loss, 0.7182439337288026);

    TEST_ASSERT_EQUAL_FLOAT(fc_data->weights->data[0], -0.2679348142022225);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->weights->data[1], 0.4848339319438247);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->weights->data[2], -0.6315295129156053);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->weights->data[3], -0.7420722005600799);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->weights->data[4], 1.3678618463245154);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->weights->data[5], -1.229933624737421);
    TEST_ASSERT_EQUAL_FLOAT(output_data->weights->data[0], 0.0766308057690951);
    TEST_ASSERT_EQUAL_FLOAT(output_data->weights->data[1], -1.004698165338342);
    TEST_ASSERT_EQUAL_FLOAT(output_data->weights->data[2], -0.1505549847294349);

    TEST_ASSERT_EQUAL_FLOAT(fc_data->biases->data[0], -1.3171855204123333e-5);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->biases->data[1], 0.0);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->biases->data[2], -2.4724469075790463e-4);
    TEST_ASSERT_EQUAL_FLOAT(output_data->biases->data[0], 0.0012500486106543);

    loss = pearl_network_train_epoch(&network, input, output);
    TEST_ASSERT_EQUAL_FLOAT(loss, 0.7150801771957616);

    TEST_ASSERT_EQUAL_FLOAT(fc_data->weights->data[0], -0.2688913302831794);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->weights->data[1], 0.4848169107220069);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->weights->data[2], -0.6315295129156053);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->weights->data[3], -0.7420722005600799);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->weights->data[4], 1.3676672556820724);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->weights->data[5], -1.2280543770573713);
    TEST_ASSERT_EQUAL_FLOAT(output_data->weights->data[0], 0.0798675160482573);
    TEST_ASSERT_EQUAL_FLOAT(output_data->weights->data[1], -1.004698165338342);
    TEST_ASSERT_EQUAL_FLOAT(output_data->weights->data[2], -0.1334351602381933);

    TEST_ASSERT_EQUAL_FLOAT(fc_data->biases->data[0], -3.0193077021853888e-5);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->biases->data[1], 0.0);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->biases->data[2], -4.4183533320092531e-4);
    TEST_ASSERT_EQUAL_FLOAT(output_data->biases->data[0], 0.0022947400840857);

    pearl_network_destroy(&network);
    pearl_tensor_destroy(&input);
    pearl_tensor_destroy(&output);
}

void test_network_regression()
{
    pearl_network *network = pearl_network_create(2, 1);
    pearl_layer *input_layer = pearl_layer_create_input(2);
    network->input_layer = input_layer;
    pearl_layer *fc = pearl_layer_create_fully_connected(3, 2);
    pearl_layer_add_child(&input_layer, &fc);
    pearl_layer *output_layer = pearl_layer_create_fully_connected(1, 3);
    output_layer->activation = pearl_activation_create(pearl_activation_type_sigmoid);
    pearl_layer_add_child(&fc, &output_layer);
    network->output_layer = output_layer;
    network->learning_rate = 0.1f;
    network->loss = pearl_loss_create(pearl_loss_mean_squared_error);

    pearl_tensor *input = pearl_tensor_create(2, 100, 2);
    pearl_tensor *output = pearl_tensor_create(2, 1, 100);
    int counter_in = 0;
    int counter_out = 0;
    float min = -5.0f;
    float max = 5.0f;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            float a = min + (rand() / (RAND_MAX / (max - min)));
            float b = min + (rand() / (RAND_MAX / (max - min)));
            input->data[counter_in] = a;
            input->data[counter_in + 1] = b;
            output->data[counter_out] = powf(a, 2.0f) + b;
            counter_in += 2;
            counter_out++;
        }
    }

    float loss;
    for (int i = 0; i < 1000; i++) {
        loss = pearl_network_train_epoch(&network, input, output);
    }
    printf("MSE Loss = %0.16f\n", loss);
    pearl_tensor *pred = pearl_network_calculate(&network, input);

    pearl_network_destroy(&network);
    pearl_tensor_destroy(&input);
    pearl_tensor_destroy(&output);
    pearl_tensor_destroy(&pred);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_network_create);
    RUN_TEST(test_network_add_layers);
    RUN_TEST(test_network_save_load);
    RUN_TEST(test_network_epoch_check);
    RUN_TEST(test_network_regression);
    return UNITY_END();
}
