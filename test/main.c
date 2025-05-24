#include "unity.h"
#include <pearl_network.h>
#include <pearl_json.h>
#include <pearl_print.h>
#include <time.h>

void setUp(void)
{
    srand(time(NULL));
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
    TEST_ASSERT_NULL(data_drop->weights);

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
    fc_data->weights->data[0] = -0.2670205f;
    fc_data->weights->data[1] = 0.4848471f;
    fc_data->weights->data[2] = -0.6315295f;
    fc_data->weights->data[3] = -0.7420722f;
    fc_data->weights->data[4] = 1.3681090f;
    fc_data->weights->data[5] = -1.2320253f;
    output_data->weights->data[0] = 0.0733915f;
    output_data->weights->data[1] = -1.0046981f;
    output_data->weights->data[2] = -0.1679168f;

    float loss = pearl_network_train_epoch(&network, input, output);
    TEST_ASSERT_EQUAL_FLOAT(loss, 0.7182439f);

    TEST_ASSERT_EQUAL_FLOAT(fc_data->weights->data[0], -0.2679348f);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->weights->data[1], 0.4848339f);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->weights->data[2], -0.6315295f);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->weights->data[3], -0.7420722f);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->weights->data[4], 1.3678618f);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->weights->data[5], -1.2299336f);
    TEST_ASSERT_EQUAL_FLOAT(output_data->weights->data[0], 0.0766308f);
    TEST_ASSERT_EQUAL_FLOAT(output_data->weights->data[1], -1.0046981f);
    TEST_ASSERT_EQUAL_FLOAT(output_data->weights->data[2], -0.1505549f);

    TEST_ASSERT_EQUAL_FLOAT(fc_data->biases->data[0], -1.3171855e-5f);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->biases->data[1], 0.0f);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->biases->data[2], -2.4724469e-4f);
    TEST_ASSERT_EQUAL_FLOAT(output_data->biases->data[0], 0.00125005f);

    loss = pearl_network_train_epoch(&network, input, output);
    TEST_ASSERT_EQUAL_FLOAT(loss, 0.7150801f);

    TEST_ASSERT_EQUAL_FLOAT(fc_data->weights->data[0], -0.2688913f);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->weights->data[1], 0.4848169f);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->weights->data[2], -0.6315295f);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->weights->data[3], -0.7420722f);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->weights->data[4], 1.3676672f);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->weights->data[5], -1.2280543f);
    TEST_ASSERT_EQUAL_FLOAT(output_data->weights->data[0], 0.0798675f);
    TEST_ASSERT_EQUAL_FLOAT(output_data->weights->data[1], -1.0046981f);
    TEST_ASSERT_EQUAL_FLOAT(output_data->weights->data[2], -0.1334351f);

    TEST_ASSERT_EQUAL_FLOAT(fc_data->biases->data[0], -3.0193077e-5f);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->biases->data[1], 0.0f);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->biases->data[2], -4.4183533e-4f);
    TEST_ASSERT_EQUAL_FLOAT(output_data->biases->data[0], 0.002294742f);

    loss = pearl_network_train_epoch(&network, input, output);
    TEST_ASSERT_EQUAL_FLOAT(loss, 0.7120114f);

    TEST_ASSERT_EQUAL_FLOAT(fc_data->weights->data[0], -0.2698902f);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->weights->data[1], 0.484796f);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->weights->data[2], -0.6315295f);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->weights->data[3], -0.7420722f);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->weights->data[4], 1.367518f);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->weights->data[5], -1.226386f);
    TEST_ASSERT_EQUAL_FLOAT(output_data->weights->data[0], 0.0831031f);
    TEST_ASSERT_EQUAL_FLOAT(output_data->weights->data[1], -1.0046981f);
    TEST_ASSERT_EQUAL_FLOAT(output_data->weights->data[2], -0.1165501f);

    TEST_ASSERT_EQUAL_FLOAT(fc_data->biases->data[0], -5.114581e-05);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->biases->data[1], 0.0f);
    TEST_ASSERT_EQUAL_FLOAT(fc_data->biases->data[2], -0.0005908558);
    TEST_ASSERT_EQUAL_FLOAT(output_data->biases->data[0], 0.003140901);

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

    float loss1;
    for (int i = 0; i < 100; i++) {
        loss1 = pearl_network_train_epoch(&network, input, output);
    }

    float loss2;
    for (int i = 0; i < 100; i++) {
        loss2 = pearl_network_train_epoch(&network, input, output);
    }

    TEST_ASSERT_LESS_THAN_FLOAT(loss1, loss2);

    pearl_tensor *pred = pearl_network_calculate(&network, input);

    pearl_network_destroy(&network);
    pearl_tensor_destroy(&input);
    pearl_tensor_destroy(&output);
    pearl_tensor_destroy(&pred);
}

// Helper function to create a simple network with one input and one dropout layer
static pearl_network* create_simple_dropout_network(unsigned int num_neurons, float dropout_rate_val) {
    pearl_network *network = pearl_network_create();
    TEST_ASSERT_NOT_NULL(network);

    pearl_layer *input_l = pearl_layer_create_input(num_neurons);
    network->input_layer = input_l;

    pearl_layer *dropout_l = pearl_layer_create_dropout(num_neurons);
    ((pearl_layer_data_dropout*)dropout_l->layer_data)->rate = dropout_rate_val;
    
    pearl_layer_add_child(&input_l, &dropout_l);
    network->output_layer = dropout_l; // Treat dropout as output for this simple test
    return network;
}

void test_dropout_layer_forward_training_mode(void) {
    unsigned int num_neurons = 1000;
    unsigned int batch_size = 1; // Test with batch size 1 for simplicity of counting
    float dropout_rate = 0.3f;
    float scale = 1.0f / (1.0f - dropout_rate);

    pearl_network *net = create_simple_dropout_network(num_neurons, dropout_rate);
    net->is_training = true;

    // Create input tensor (e.g., all ones)
    pearl_tensor *input_data = pearl_tensor_create(2, num_neurons, batch_size);
    for (unsigned int i = 0; i < num_neurons * batch_size; ++i) {
        input_data->data[i] = 1.0f;
    }

    // Set input layer's activation directly for the test
    // Normally pearl_network_forward would handle this, but for isolated layer test:
    if (net->input_layer->a == NULL) {
         net->input_layer->a = pearl_tensor_create(2, num_neurons, batch_size);
    }
    pearl_tensor_destroy(&net->input_layer->a); // Destroy if pre-existing from create_simple
    net->input_layer->a = pearl_tensor_copy(input_data);


    // Perform forward pass focusing on the dropout layer
    // The actual network forward will call layer_forward
    pearl_layer_forward(&net->input_layer, &net->input_layer->child_layers[0], net->is_training);
    
    pearl_layer *dropout_layer = net->input_layer->child_layers[0];
    pearl_tensor *output_a = dropout_layer->a;
    pearl_layer_data_dropout *dropout_data_cast = (pearl_layer_data_dropout*)dropout_layer->layer_data;
    pearl_tensor *mask = dropout_data_cast->weights;

    TEST_ASSERT_NOT_NULL(output_a);
    TEST_ASSERT_NOT_NULL(mask);
    TEST_ASSERT_EQUAL_UINT(output_a->dimension, 2);
    TEST_ASSERT_EQUAL_UINT(output_a->size[0], num_neurons);
    TEST_ASSERT_EQUAL_UINT(output_a->size[1], batch_size);
    TEST_ASSERT_EQUAL_UINT(mask->dimension, output_a->dimension);
    TEST_ASSERT_EQUAL_UINT(mask->size[0], output_a->size[0]);
    TEST_ASSERT_EQUAL_UINT(mask->size[1], output_a->size[1]);


    unsigned int zero_count = 0;
    for (unsigned int i = 0; i < num_neurons * batch_size; ++i) {
        if (mask->data[i] == 0.0f) {
            zero_count++;
            TEST_ASSERT_EQUAL_FLOAT(0.0f, output_a->data[i]);
        } else {
            TEST_ASSERT_EQUAL_FLOAT(1.0f, mask->data[i]);
            TEST_ASSERT_EQUAL_FLOAT(input_data->data[i] * scale, output_a->data[i]);
        }
    }

    // Check if the number of zeros is roughly rate * total_elements
    // This is a probabilistic test, so allow some tolerance.
    float expected_zeros = dropout_rate * num_neurons * batch_size;
    float tolerance = 0.1f * num_neurons * batch_size; // 10% tolerance
    TEST_ASSERT_FLOAT_WITHIN(tolerance, expected_zeros, (float)zero_count);
    
    pearl_tensor_destroy(&input_data);
    pearl_network_destroy(&net);
}

void test_dropout_layer_forward_inference_mode(void) {
    unsigned int num_neurons = 100;
    unsigned int batch_size = 2;
    float dropout_rate = 0.5f;

    pearl_network *net = create_simple_dropout_network(num_neurons, dropout_rate);
    net->is_training = false;

    pearl_tensor *input_data = pearl_tensor_create(2, num_neurons, batch_size);
    for (unsigned int i = 0; i < num_neurons * batch_size; ++i) {
        input_data->data[i] = (float)(i + 1); // Some varied data
    }
    
    // Set input layer's activation
    if (net->input_layer->a == NULL) {
         net->input_layer->a = pearl_tensor_create(2, num_neurons, batch_size);
    }
    pearl_tensor_destroy(&net->input_layer->a);
    net->input_layer->a = pearl_tensor_copy(input_data);

    pearl_layer_forward(&net->input_layer, &net->input_layer->child_layers[0], net->is_training);

    pearl_tensor *output_a = net->input_layer->child_layers[0]->a;
    TEST_ASSERT_NOT_NULL(output_a);

    for (unsigned int i = 0; i < num_neurons * batch_size; ++i) {
        TEST_ASSERT_EQUAL_FLOAT(input_data->data[i], output_a->data[i]);
    }
    
    pearl_tensor_destroy(&input_data);
    pearl_network_destroy(&net);
}

void test_dropout_layer_backward_pass(void) {
    unsigned int num_neurons = 100;
    unsigned int batch_size = 1;
    float dropout_rate = 0.4f;
    float scale = 1.0f / (1.0f - dropout_rate);

    pearl_network *net = create_simple_dropout_network(num_neurons, dropout_rate);
    net->is_training = true;

    pearl_tensor *input_val = pearl_tensor_create(2, num_neurons, batch_size);
    for(unsigned int i=0; i < num_neurons * batch_size; ++i) input_val->data[i] = 1.0f;
    
    if (net->input_layer->a == NULL) {
        net->input_layer->a = pearl_tensor_create(2, num_neurons, batch_size);
    }
    pearl_tensor_destroy(&net->input_layer->a);
    net->input_layer->a = pearl_tensor_copy(input_val);


    // Forward pass to generate the mask
    pearl_layer_forward(&net->input_layer, &net->input_layer->child_layers[0], net->is_training);

    pearl_layer* dropout_layer = net->input_layer->child_layers[0];
    pearl_layer_data_dropout *dropout_data_cast = (pearl_layer_data_dropout*)dropout_layer->layer_data;
    pearl_tensor *mask = dropout_data_cast->weights;
    TEST_ASSERT_NOT_NULL(mask);

    // Manually set incoming gradients for the dropout layer (da of dropout layer)
    if (dropout_layer->da == NULL) {
        dropout_layer->da = pearl_tensor_create(2, num_neurons, batch_size);
    }
    for (unsigned int i = 0; i < num_neurons * batch_size; ++i) {
        dropout_layer->da->data[i] = (float)(i + 1); // Arbitrary gradient values
    }

    // Perform backward pass for the dropout layer
    pearl_layer_backward(&dropout_layer, &net->input_layer); // child_layer, parent_layer

    pearl_tensor *parent_da = net->input_layer->da; // This is da for the layer *before* dropout
    TEST_ASSERT_NOT_NULL(parent_da);

    for (unsigned int i = 0; i < num_neurons * batch_size; ++i) {
        float expected_grad = dropout_layer->da->data[i] * mask->data[i] * scale;
        TEST_ASSERT_EQUAL_FLOAT(expected_grad, parent_da->data[i]);
    }

    pearl_tensor_destroy(&input_val);
    pearl_network_destroy(&net);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_network_create);
    RUN_TEST(test_network_add_layers);
    RUN_TEST(test_network_save_load);
    RUN_TEST(test_network_epoch_check);
    RUN_TEST(test_network_regression);
    RUN_TEST(test_dropout_layer_forward_training_mode);
    RUN_TEST(test_dropout_layer_forward_inference_mode);
    RUN_TEST(test_dropout_layer_backward_pass);
    return UNITY_END();
}
