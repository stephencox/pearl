#include <QString>
#include <QtTest>
extern "C"
{
#include <pearl_network.h>
}

class PearlLibTestNetwork : public QObject
{
        Q_OBJECT

    public:
        PearlLibTestNetwork();

    private Q_SLOTS:
        void testCaseCreateNetwork();
        void testCaseAddLayer();
};

PearlLibTestNetwork::PearlLibTestNetwork()
{
}

void PearlLibTestNetwork::testCaseCreateNetwork()
{
    pearl_network *network = NULL;
    QVERIFY(network == NULL);
    network = pearl_network_create(2, 1);
    QVERIFY(network != NULL);
    QVERIFY(network->num_input == 2);
    QVERIFY(network->num_output == 1);
    QVERIFY(network->learning_rate == 1e-3);
    QVERIFY(network->loss == pearl_loss_binary_cross_entropy);
    QVERIFY(network->optimiser == pearl_optimiser_sgd);
    QVERIFY(network->version.major == PEARL_NETWORK_VERSION_MAJOR);
    QVERIFY(network->version.minor == PEARL_NETWORK_VERSION_MINOR);
    QVERIFY(network->version.revision == PEARL_NETWORK_VERSION_REVISION);
    pearl_network_layer_add_fully_connect(network, 3, pearl_activation_function_type_relu);
    pearl_network_layer_add_output(network, 1, pearl_activation_function_type_sigmoid);
    pearl_network_layers_initialise(network);
    pearl_network_destroy(&network);
    QVERIFY(network == NULL);
}

PearlLibTestNetwork::testCaseAddLayer()
{
    pearl_network *network = NULL;
    QVERIFY(network == NULL);
    network = pearl_network_create(2, 1);
    QVERIFY(network != NULL);
    pearl_network_layer_add_fully_connect(network, 3, pearl_activation_function_type_relu);
    QVERIFY(network->num_layers == 1);
    QVERIFY(network->layers[0] != NULL);
    QVERIFY(network->layers[0]->activation_function == pearl_activation_function_type_relu);
    QVERIFY(network->layers[0]->dropout_rate == 0.0);
    QVERIFY(network->layers[0]->neurons == 3);
    QVERIFY(network->layers[0]->type == pearl_layer_type_fully_connect);
    QVERIFY(network->layers[0]->version.major == PEARL_LAYER_VERSION_MAJOR);
    QVERIFY(network->layers[0]->version.minor == PEARL_LAYER_VERSION_MINOR);
    QVERIFY(network->layers[0]->version.revision == PEARL_LAYER_VERSION_REVISION);
    QVERIFY(network->layers[0]->weights == NULL);
    QVERIFY(network->layers[0]->biases == NULL);
    pearl_network_layers_initialise(network);
    QVERIFY(network->layers[0]->weights != NULL);
    QVERIFY(network->layers[0]->weights->dimension == 2);
    QVERIFY(network->layers[0]->weights->size[0] == network->num_input);
    QVERIFY(network->layers[0]->weights->size[1] == 3);
    QVERIFY(network->layers[0]->biases != NULL);
    QVERIFY(network->layers[0]->biases->dimension == 1);
    QVERIFY(network->layers[0]->biases->size[0] == 3);
    pearl_network_destroy(&network);
    QVERIFY(network == NULL);
}

QTEST_APPLESS_MAIN(PearlLibTestTest)

#include "tst_pearllibtesttest.moc"
