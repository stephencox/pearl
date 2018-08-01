#include <QString>
#include <QtTest>
extern "C"
{
#include <pearl_network.h>
#include <pearl_json.h>
}

class PearlLibTestNetwork : public QObject
{
        Q_OBJECT

    private Q_SLOTS:
        void testCaseCreateNetwork();
        void testCaseAddLayer();
        void testCaseSaveLoad();
        void testCaseNetworkForwardCheck();
};

void PearlLibTestNetwork::testCaseCreateNetwork()
{
    pearl_network *network = NULL;
    QVERIFY(network == NULL);
    network = pearl_network_create(2, 1);
    QVERIFY(network != NULL);
    QCOMPARE(network->num_layers, 0);
    QCOMPARE(network->num_input, 2);
    QCOMPARE(network->num_output, 1);
    QCOMPARE(network->learning_rate, 1e-3);
    QCOMPARE(network->loss_type, pearl_loss_binary_cross_entropy);
    QCOMPARE(network->optimiser, pearl_optimiser_sgd);
    QCOMPARE(network->version.major, PEARL_NETWORK_VERSION_MAJOR);
    QCOMPARE(network->version.minor, PEARL_NETWORK_VERSION_MINOR);
    QCOMPARE(network->version.revision, PEARL_NETWORK_VERSION_REVISION);
    pearl_network_destroy(&network);
    QVERIFY(network == NULL);
}

void PearlLibTestNetwork::testCaseAddLayer()
{
    pearl_network *network = pearl_network_create(10, 1);

    pearl_network_layer_add_fully_connect(&network, 5, pearl_activation_function_type_relu);

    QCOMPARE(network->num_layers, 1);
    QVERIFY(network->layers[0] != NULL);
    QCOMPARE(network->layers[0]->activation_function, pearl_activation_function_type_relu);
    //QCOMPARE(network->layers[0]->dropout_rate, 0.0);
    QCOMPARE(network->layers[0]->neurons, 5);
    QCOMPARE(network->layers[0]->type, pearl_layer_type_fully_connect);
    QCOMPARE(network->layers[0]->version.major, PEARL_LAYER_VERSION_MAJOR);
    QCOMPARE(network->layers[0]->version.minor, PEARL_LAYER_VERSION_MINOR);
    QCOMPARE(network->layers[0]->version.revision, PEARL_LAYER_VERSION_REVISION);
    QVERIFY(network->layers[0]->weights == NULL);
    QVERIFY(network->layers[0]->biases == NULL);

    pearl_network_layer_add_output(&network, pearl_activation_function_type_sigmoid);

    QCOMPARE(network->num_layers, 2);
    QVERIFY(network->layers[1] != NULL);
    QCOMPARE(network->layers[1]->activation_function, pearl_activation_function_type_sigmoid);
    //QCOMPARE(network->layers[1]->dropout_rate, 0.0);
    QCOMPARE(network->layers[1]->neurons, 1);
    QCOMPARE(network->layers[1]->type, pearl_layer_type_fully_connect);
    QCOMPARE(network->layers[1]->version.major, PEARL_LAYER_VERSION_MAJOR);
    QCOMPARE(network->layers[1]->version.minor, PEARL_LAYER_VERSION_MINOR);
    QCOMPARE(network->layers[1]->version.revision, PEARL_LAYER_VERSION_REVISION);
    QVERIFY(network->layers[1]->weights == NULL);
    QVERIFY(network->layers[1]->biases == NULL);

    pearl_network_layers_initialise(&network);

    QVERIFY(network->layers[0]->weights != NULL);
    QCOMPARE(network->layers[0]->weights->dimension, 2);
    QCOMPARE(network->layers[0]->weights->size[0], 5);
    QCOMPARE(network->layers[0]->weights->size[1], 10);
    QVERIFY(network->layers[0]->biases != NULL);
    QCOMPARE(network->layers[0]->biases->dimension, 1);
    QCOMPARE(network->layers[0]->biases->size[0], 5);

    QVERIFY(network->layers[1]->weights != NULL);
    QCOMPARE(network->layers[1]->weights->dimension, 2);
    QCOMPARE(network->layers[1]->weights->size[0], 1);
    QCOMPARE(network->layers[1]->weights->size[1], 5);
    QVERIFY(network->layers[1]->biases != NULL);
    QCOMPARE(network->layers[1]->biases->dimension, 1);
    QCOMPARE(network->layers[1]->biases->size[0], 1);

    pearl_network_destroy(&network);
}

void PearlLibTestNetwork::testCaseSaveLoad()
{
    pearl_network *network = pearl_network_create(10, 1);
    pearl_network_layer_add_fully_connect(&network, 5, pearl_activation_function_type_relu);
    pearl_network_layer_add_output(&network, pearl_activation_function_type_sigmoid);
    pearl_network_layers_initialise(&network);
    pearl_network_save("network.json", network);
    pearl_network_destroy(&network);

    pearl_network *network_load = pearl_network_load("network.json");
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

    pearl_network_destroy(&network_load);
}

void PearlLibTestNetwork::testCaseNetworkForwardCheck()
{
    pearl_network *network = pearl_network_create(2, 1);
    pearl_network_layer_add_fully_connect(&network, 3, pearl_activation_function_type_relu);
    pearl_network_layer_add_output(&network, pearl_activation_function_type_sigmoid);
    pearl_network_layers_initialise(&network);

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

    pearl_network_train_epoch(&network, input, output);

    pearl_network_destroy(&network);
    pearl_tensor_destroy(&input);
    pearl_tensor_destroy(&output);
}

QTEST_APPLESS_MAIN(PearlLibTestNetwork)

#include "tst_pearllib_network.moc"
