#include <QString>
#include <QtTest>
extern "C"
{
#include <pearl_network.h>
}

class PearlLibTestTest : public QObject
{
        Q_OBJECT

    public:
        PearlLibTestTest();

    private Q_SLOTS:
        void testCaseCreateNetwork();
};

PearlLibTestTest::PearlLibTestTest()
{
}

void PearlLibTestTest::testCaseCreateNetwork()
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

QTEST_APPLESS_MAIN(PearlLibTestTest)

#include "tst_pearllibtesttest.moc"
