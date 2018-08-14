#include <munit.h>
#include <pearl_network.h>
#include <pearl_json.h>
#include <pearl_print.h>

static MunitResult test_network_create(const MunitParameter params[], void *data)
{
    (void) params;
    (void) data;
    pearl_network *network = pearl_network_create();
    munit_assert_not_null(network);
    munit_assert_null(network->input_layer);
    munit_assert_double(network->learning_rate, ==, 1e-3);
    munit_assert_int(network->loss.type, ==, pearl_loss_binary_cross_entropy);
    munit_assert_int(network->optimiser, ==, pearl_optimiser_sgd);
    munit_assert_uint(network->version.major, ==, PEARL_NETWORK_VERSION_MAJOR);
    munit_assert_uint(network->version.minor, ==, PEARL_NETWORK_VERSION_MINOR);
    munit_assert_uint(network->version.revision, ==, PEARL_NETWORK_VERSION_REVISION);
    pearl_network_destroy(&network);
    munit_assert_null(network);
    return MUNIT_OK;
}

static MunitResult test_network_add_layers(const MunitParameter params[], void *data)
{
    (void) params;
    (void) data;
    pearl_network *network = pearl_network_create();
    pearl_layer *input = pearl_layer_create_input(5);
    network->input_layer = input;
    munit_assert_int(input->type, ==, pearl_layer_type_input);
    munit_assert_int(input->num_child_layers, ==, 0);
    munit_assert_null(input->child_layers);
    munit_assert_int(input->num_parent_layers, ==, 0);
    munit_assert_null(input->parent_layers);
    munit_assert_not_null(input->layer_data);
    pearl_layer_data_input *data_input = (pearl_layer_data_input *)input->layer_data;
    munit_assert_int(data_input->num_neurons, ==, 5);
    munit_assert_int(data_input->activation.type, ==, pearl_activation_function_type_linear);

    pearl_layer *drop = pearl_layer_create_dropout(5);
    pearl_layer_add_child(&input, &drop);
    munit_assert_int(input->num_parent_layers, ==, 0);
    munit_assert_null(input->parent_layers);
    munit_assert_int(input->num_child_layers, ==, 1);
    munit_assert_not_null(input->child_layers);
    munit_assert_not_null(input->child_layers[0]);
    munit_assert_ptr_equal(input->child_layers[0], drop);
    munit_assert_int(drop->type, ==, pearl_layer_type_dropout);
    munit_assert_int(drop->num_child_layers, ==, 0);
    munit_assert_null(drop->child_layers);
    munit_assert_int(drop->num_parent_layers, ==, 1);
    munit_assert_not_null(drop->parent_layers);
    munit_assert_not_null(drop->parent_layers[0]);
    munit_assert_int(drop->parent_layers[0]->type, ==, pearl_layer_type_input);
    munit_assert_not_null(drop->layer_data);
    pearl_layer_data_dropout *data_drop = (pearl_layer_data_dropout *)drop->layer_data;
    munit_assert_int(data_drop->num_neurons, ==, 5);
    munit_assert_double(data_drop->rate, ==, 0.5);
    munit_assert_int(data_drop->weights->dimension, ==, 1);
    munit_assert_not_null(data_drop->weights->size);
    munit_assert_int(data_drop->weights->size[0], ==, 5);
    munit_assert_not_null(data_drop->weights->data);
    for (unsigned int i = 0; i < data_drop->weights->size[0]; i++) {
        munit_assert_double(data_drop->weights->data[i], ==, 0.0);
    }

    pearl_layer *fc = pearl_layer_create_fully_connected(5, 1);
    pearl_layer_add_child(&drop, &fc);
    munit_assert_int(drop->num_child_layers, ==, 1);
    munit_assert_not_null(drop->child_layers);
    munit_assert_not_null(drop->child_layers[0]);
    munit_assert_ptr_equal(drop->child_layers[0], fc);
    munit_assert_int(fc->type, ==, pearl_layer_type_fully_connected);
    munit_assert_int(fc->num_child_layers, ==, 0);
    munit_assert_null(fc->child_layers);
    munit_assert_int(fc->num_parent_layers, ==, 1);
    munit_assert_not_null(fc->parent_layers);
    munit_assert_not_null(fc->parent_layers[0]);
    munit_assert_int(fc->parent_layers[0]->type, ==, pearl_layer_type_dropout);
    munit_assert_not_null(fc->layer_data);
    pearl_layer_data_fully_connected *data_fc = (pearl_layer_data_fully_connected *)fc->layer_data;
    munit_assert_int(data_fc->num_neurons, ==, 5);
    munit_assert_int(data_fc->activation.type, ==, pearl_activation_function_type_relu);
    munit_assert_not_null(data_fc->weights);
    munit_assert_int(data_fc->weights->dimension, ==, 2);
    munit_assert_not_null(data_fc->weights->size);
    munit_assert_int(data_fc->weights->size[0], ==, 5);
    munit_assert_int(data_fc->weights->size[1], ==, 1);
    munit_assert_not_null(data_fc->weights->data);
    for (unsigned int i = 0; i < data_fc->weights->size[0]; i++) {
        munit_assert_double(data_fc->weights->data[i], !=, 0.0);
    }
    munit_assert_not_null(data_fc->biases);
    munit_assert_int(data_fc->biases->dimension, ==, 1);
    munit_assert_not_null(data_fc->biases->size);
    munit_assert_int(data_fc->biases->size[0], ==, 5);
    munit_assert_not_null(data_fc->biases->data);
    for (unsigned int i = 0; i < data_fc->biases->size[0]; i++) {
        munit_assert_int(data_fc->biases->data[i], ==, 0.0);
    }

    pearl_layer *output = pearl_layer_create_output(1);
    pearl_layer_add_child(&fc, &output);
    munit_assert_int(fc->num_child_layers, ==, 1);
    munit_assert_not_null(fc->child_layers);
    munit_assert_not_null(fc->child_layers[0]);
    munit_assert_ptr_equal(fc->child_layers[0], output);
    munit_assert_int(output->type, ==, pearl_layer_type_output);
    munit_assert_int(output->num_child_layers, ==, 0);
    munit_assert_null(output->child_layers);
    munit_assert_int(output->num_parent_layers, ==, 1);
    munit_assert_not_null(output->parent_layers);
    munit_assert_not_null(output->parent_layers[0]);
    munit_assert_int(output->parent_layers[0]->type, ==, pearl_layer_type_fully_connected);
    munit_assert_not_null(output->layer_data);
    munit_assert_not_null(output->layer_data);
    pearl_layer_data_input *data_output = (pearl_layer_data_input *)output->layer_data;
    munit_assert_int(data_output->num_neurons, ==, 1);
    munit_assert_int(data_output->activation.type, ==, pearl_activation_function_type_linear);

    pearl_network_destroy(&network);
    munit_assert_null(network);
    return MUNIT_OK;
}

static MunitResult test_network_save_load(const MunitParameter params[], void *data)
{
    (void) params;
    (void) data;
    /*pearl_network *network = pearl_network_create(10, 1);
    pearl_network_layer_add_fully_connect(&network, 5, pearl_activation_function_type_relu);
    pearl_network_layer_add_output(&network, pearl_activation_function_type_sigmoid);
    pearl_network_layers_initialise(&network);
    pearl_json_network_serialise("network.json", network);
    pearl_network_destroy(&network);

    pearl_network *network_load = pearl_json_network_deserialise("network.json");
    QVERIFY(network_load != NULL);
    QVERIFY(network_load->layers[0]->weights != NULL);
    QCOMPARE(network_load->layers[0]->weights->dimension, 2);
    QCOMPARE(network_load->layers[0]->weights->size[0], 5);
    QCOMPARE(network_load->layers[0]->weights->size[1], 10);
    QVERIFY(network_load->layers[0]->biases != NULL);
    QCOMPARE(network_load->layers[0]->biases->dimension, 1);
    QCOMPARE(network_load->layers[0]->biases->size[0], 5);

    QVERIFY(network_load->layers[1]->weights != NULL);
    QCOMPARE(network_load->layers[1]->weights->dimension, 2);
    QCOMPARE(network_load->layers[1]->weights->size[0], 1);
    QCOMPARE(network_load->layers[1]->weights->size[1], 5);
    QVERIFY(network_load->layers[1]->biases != NULL);
    QCOMPARE(network_load->layers[1]->biases->dimension, 1);
    QCOMPARE(network_load->layers[1]->biases->size[0], 1);

    pearl_network_destroy(&network_load);*/
    return MUNIT_OK;
}

static MunitResult test_network_epoch_check(const MunitParameter params[], void *data)
{
    (void) params;
    (void) data;
    /*pearl_network *network = pearl_network_create(2, 1);
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

    network->layers[0]->weights->data[0] = -0.2670205687292891;
    network->layers[0]->weights->data[1] = 0.4848471037990288;
    network->layers[0]->weights->data[2] = -0.6315295129156053;
    network->layers[0]->weights->data[3] = -0.7420722005600799;
    network->layers[0]->weights->data[4] = 1.3681090910152733;
    network->layers[0]->weights->data[5] = -1.232025382027168;
    network->layers[1]->weights->data[0] = 0.0733915213686632;
    network->layers[1]->weights->data[1] = -1.004698165338342;
    network->layers[1]->weights->data[2] = -0.1679168826901067;

    double loss = pearl_network_train_epoch(&network, input, output);
    QVERIFY(qAbs(loss - 0.7182439337288026) < 1e-15);

    QVERIFY(qAbs(network->layers[0]->weights->data[0] + 0.2679348142022225) < 1e-15);
    QVERIFY(qAbs(network->layers[0]->weights->data[1] - 0.4848339319438247) < 1e-15);
    QVERIFY(qAbs(network->layers[0]->weights->data[2] + 0.6315295129156053) < 1e-15);
    QVERIFY(qAbs(network->layers[0]->weights->data[3] + 0.7420722005600799) < 1e-15);
    QVERIFY(qAbs(network->layers[0]->weights->data[4] - 1.3678618463245154) < 1e-15);
    QVERIFY(qAbs(network->layers[0]->weights->data[5] + 1.229933624737421) < 1e-15);
    QVERIFY(qAbs(network->layers[1]->weights->data[0] - 0.0766308057690951) < 1e-15);
    QVERIFY(qAbs(network->layers[1]->weights->data[1] + 1.004698165338342) < 1e-15);
    QVERIFY(qAbs(network->layers[1]->weights->data[2] + 0.1505549847294349) < 1e-15);

    QVERIFY(qAbs(network->layers[0]->biases->data[0] + 1.3171855204123333e-5) < 1e-15);
    QVERIFY(qAbs(network->layers[0]->biases->data[1] - 0.0) < 1e-15);
    QVERIFY(qAbs(network->layers[0]->biases->data[2] + 2.4724469075790463e-4) < 1e-15);
    QVERIFY(qAbs(network->layers[1]->biases->data[0] - 0.0012500486106543) < 1e-15);

    loss = pearl_network_train_epoch(&network, input, output);
    QVERIFY(qAbs(loss - 0.7150801771957616) < 1e-15);

    QVERIFY(qAbs(network->layers[0]->weights->data[0] + 0.2688913302831794) < 1e-15);
    QVERIFY(qAbs(network->layers[0]->weights->data[1] - 0.4848169107220069) < 1e-15);
    QVERIFY(qAbs(network->layers[0]->weights->data[2] + 0.6315295129156053) < 1e-15);
    QVERIFY(qAbs(network->layers[0]->weights->data[3] + 0.7420722005600799) < 1e-15);
    QVERIFY(qAbs(network->layers[0]->weights->data[4] - 1.3676672556820724) < 1e-15);
    QVERIFY(qAbs(network->layers[0]->weights->data[5] + 1.2280543770573713) < 1e-15);
    QVERIFY(qAbs(network->layers[1]->weights->data[0] - 0.0798675160482573) < 1e-15);
    QVERIFY(qAbs(network->layers[1]->weights->data[1] + 1.004698165338342) < 1e-15);
    QVERIFY(qAbs(network->layers[1]->weights->data[2] + 0.1334351602381933) < 1e-15);

    QVERIFY(qAbs(network->layers[0]->biases->data[0] + 3.0193077021853888e-5) < 1e-15);
    QVERIFY(qAbs(network->layers[0]->biases->data[1] - 0.0) < 1e-15);
    QVERIFY(qAbs(network->layers[0]->biases->data[2] + 4.4183533320092531e-4) < 1e-15);
    QVERIFY(qAbs(network->layers[1]->biases->data[0] - 0.0022947400840857) < 1e-15);

    pearl_network_destroy(&network);
    pearl_tensor_destroy(&input);
    pearl_tensor_destroy(&output);*/
    return MUNIT_OK;
}

static MunitResult test_network_regression(const MunitParameter params[], void *data)
{
    (void) params;
    (void) data;
    /*pearl_network *network = pearl_network_create(2, 1);
    pearl_network_layer_add_fully_connect(&network, 3, pearl_activation_function_type_relu);
    pearl_network_layer_add_output(&network, pearl_activation_function_type_linear);
    pearl_network_layers_initialise(&network);
    network->learning_rate = 0.1;
    network->loss = pearl_loss_create(pearl_loss_mean_squared_error);

    pearl_tensor *input = pearl_tensor_create(2, 100, 2);
    pearl_tensor *output = pearl_tensor_create(2, 1, 100);
    int counter_in = 0, counter_out = 0;
    double min = -5.0;
    double max = 5.0;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            double a = min + (rand() / (RAND_MAX / (max - min)));
            double b = min + (rand() / (RAND_MAX / (max - min)));
            input->data[counter_in] = a;
            input->data[counter_in + 1] = b;
            output->data[counter_out] = pow(a, 2) + b;
            counter_in += 2;
            counter_out++;
        }
    }

    double loss;
    for (int i = 0; i < 1000; i++) {
        loss = pearl_network_train_epoch(&network, input, output);
    }
    printf("MSE Loss = %0.16f\n", loss);
    pearl_tensor *pred = pearl_network_calculate(&network, input);

    pearl_network_destroy(&network);
    pearl_tensor_destroy(&input);
    pearl_tensor_destroy(&output);
    pearl_tensor_destroy(&pred);*/
    return MUNIT_OK;
}

static MunitTest test_suite_tests[] = {
    {
        (char *) "/network/create",
        test_network_create,
        NULL,
        NULL,
        MUNIT_TEST_OPTION_NONE,
        NULL
    },
    {
        (char *) "/network/add_layers",
        test_network_add_layers,
        NULL,
        NULL,
        MUNIT_TEST_OPTION_NONE,
        NULL
    },
    { NULL, NULL, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL }
};

static const MunitSuite test_suite = {
    (char *) "/tests",
    test_suite_tests,
    NULL,
    1,
    MUNIT_SUITE_OPTION_NONE
};

int main(int argc, char *argv[MUNIT_ARRAY_PARAM(argc + 1)])
{
    return munit_suite_main(&test_suite, NULL, argc, argv);
    return 0;
}
