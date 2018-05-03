#include <QString>
#include <QtTest>
extern "C"
{
#include <pearl_network.h>
}

class PearlLibTestNetwork : public QObject
{
        Q_OBJECT

    private Q_SLOTS:
        void testCaseCreateNetwork();
        void testCaseAddLayer();
};

void PearlLibTestNetwork::testCaseCreateNetwork()
{
    pearl_network *network = NULL;
    QVERIFY(network == NULL);
    network = pearl_network_create(2, 1);
    QVERIFY(network != NULL);
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
    pearl_network *network = NULL;
    QVERIFY(network == NULL);
    network = pearl_network_create(2, 1);
    QVERIFY(network != NULL);
    pearl_network_layer_add_fully_connect(&network, 3, pearl_activation_function_type_relu);
    QCOMPARE(network->num_layers, 1);
    QVERIFY(network->layers[0] != NULL);
    QCOMPARE(network->layers[0]->activation_function, pearl_activation_function_type_relu);
    QCOMPARE(network->layers[0]->dropout_rate, 0.0);
    QCOMPARE(network->layers[0]->neurons, 3);
    QCOMPARE(network->layers[0]->type, pearl_layer_type_fully_connect);
    QCOMPARE(network->layers[0]->version.major, PEARL_LAYER_VERSION_MAJOR);
    QCOMPARE(network->layers[0]->version.minor, PEARL_LAYER_VERSION_MINOR);
    QCOMPARE(network->layers[0]->version.revision, PEARL_LAYER_VERSION_REVISION);
    QVERIFY(network->layers[0]->weights == NULL);
    QVERIFY(network->layers[0]->biases == NULL);
    pearl_network_layers_initialise(&network);
    QVERIFY(network->layers[0]->weights != NULL);
    QCOMPARE(network->layers[0]->weights->dimension, 2);
    QCOMPARE(network->layers[0]->weights->size[0], network->num_output);
    QCOMPARE(network->layers[0]->weights->size[1], 3);
    QVERIFY(network->layers[0]->biases != NULL);
    QCOMPARE(network->layers[0]->biases->dimension, 1);
    QCOMPARE(network->layers[0]->biases->size[0], 3);
    pearl_network_destroy(&network);
    QVERIFY(network == NULL);
}

QTEST_APPLESS_MAIN(PearlLibTestNetwork)

#include "tst_pearllib_network.moc"
