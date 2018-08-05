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
    QCOMPARE(network->loss.type, pearl_loss_binary_cross_entropy);
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

    pearl_network_destroy(&network_load);
}

void PearlLibTestNetwork::testCaseNetworkForwardCheck()
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
    pearl_tensor_destroy(&output);
}

QTEST_APPLESS_MAIN(PearlLibTestNetwork)

#include "tst_pearllib_network.moc"
