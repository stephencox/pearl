#include <QString>
#include <QtTest>
extern "C"
{
#include <pearl_network.h>
#include <pearl_json.h>
#include <pearl_print.h>
}

class PearlLibTestNetwork : public QObject
{
        Q_OBJECT

    private Q_SLOTS:
        void testCaseCreateNetwork();
        void testCaseAddLayer();
        void testCaseSaveLoad();
        void testCaseNetworkForwardCheck();
        void testCaseNetworkClassification();
        void testCaseNetworkRegression();
};

void PearlLibTestNetwork::testCaseCreateNetwork()
{
    pearl_network *network = NULL;
    QVERIFY(network == NULL);
    network = pearl_network_create(1);
    QVERIFY(network != NULL);
    QCOMPARE(network->num_input_layers, 1);
    QVERIFY(network->input_layers != NULL);
    QVERIFY(network->input_layers[0] == NULL);
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
    pearl_network *network = pearl_network_create(1);
    pearl_layer *input = pearl_layer_create_input(5);
    network->input_layers[0] = input;
    QVERIFY(network->input_layers[0] == input);
    QCOMPARE(input->type, pearl_layer_type_input);
    QCOMPARE(input->num_child_layers, 0);
    QVERIFY(input->child_layers == NULL);
    QCOMPARE(input->num_parent_layers, 0);
    QVERIFY(input->parent_layers == NULL);
    QVERIFY(input->layer_data != NULL);
    pearl_layer_data_input *data_input = (pearl_layer_data_input *)input->layer_data;
    QCOMPARE(data_input->num_neurons, 5);
    QCOMPARE(data_input->activation_function, pearl_activation_function_type_linear);

    pearl_layer *drop = pearl_layer_create_dropout(5);
    pearl_layer_add_child(&input, &drop);
    QCOMPARE(input->num_parent_layers, 0);
    QVERIFY(input->parent_layers == NULL);
    QCOMPARE(input->num_child_layers, 1);
    QVERIFY(input->child_layers != NULL);
    QVERIFY(input->child_layers[0] != NULL);
    QVERIFY(input->child_layers[0] == drop);
    QCOMPARE(drop->type, pearl_layer_type_dropout);
    QCOMPARE(drop->num_child_layers, 0);
    QVERIFY(drop->child_layers == NULL);
    QCOMPARE(drop->num_parent_layers, 1);
    QVERIFY(drop->parent_layers != NULL);
    QVERIFY(drop->parent_layers[0] != NULL);
    QCOMPARE(drop->parent_layers[0]->type, pearl_layer_type_input);
    QVERIFY(drop->layer_data != NULL);
    pearl_layer_data_dropout *data_drop = (pearl_layer_data_dropout *)drop->layer_data;
    QCOMPARE(data_drop->num_neurons, 5);
    QCOMPARE(data_drop->rate, 0.5);
    QCOMPARE(data_drop->weights->dimension, 1);
    QVERIFY(data_drop->weights->size != NULL);
    QCOMPARE(data_drop->weights->size[0], 5);
    QVERIFY(data_drop->weights->data != NULL);
    for (unsigned int i = 0; i < data_drop->weights->size[0]; i++) {
        QCOMPARE(data_drop->weights->data[i], 0.0);
    }

    pearl_layer *fc = pearl_layer_create_fully_connected(5, 1);
    pearl_layer_add_child(&drop, &fc);
    QCOMPARE(drop->num_child_layers, 1);
    QVERIFY(drop->child_layers != NULL);
    QVERIFY(drop->child_layers[0] != NULL);
    QVERIFY(drop->child_layers[0] == fc);
    QCOMPARE(fc->type, pearl_layer_type_fully_connected);
    QCOMPARE(fc->num_child_layers, 0);
    QVERIFY(fc->child_layers == NULL);
    QCOMPARE(fc->num_parent_layers, 1);
    QVERIFY(fc->parent_layers != NULL);
    QVERIFY(fc->parent_layers[0] != NULL);
    QCOMPARE(fc->parent_layers[0]->type, pearl_layer_type_dropout);
    QVERIFY(fc->layer_data != NULL);
    pearl_layer_data_fully_connected *data_fc = (pearl_layer_data_fully_connected *)fc->layer_data;
    QCOMPARE(data_fc->num_neurons, 5);
    QCOMPARE(data_fc->activation_function, pearl_activation_function_type_relu);
    QVERIFY(data_fc->weights != NULL);
    QCOMPARE(data_fc->weights->dimension, 2);
    QVERIFY(data_fc->weights->size != NULL);
    QCOMPARE(data_fc->weights->size[0], 5);
    QCOMPARE(data_fc->weights->size[1], 1);
    QVERIFY(data_fc->weights->data != NULL);
    for (unsigned int i = 0; i < data_fc->weights->size[0]; i++) {
        QVERIFY(data_fc->weights->data[i] != 0.0);
    }
    QVERIFY(data_fc->biases != NULL);
    QCOMPARE(data_fc->biases->dimension, 1);
    QVERIFY(data_fc->biases->size != NULL);
    QCOMPARE(data_fc->biases->size[0], 5);
    QVERIFY(data_fc->biases->data != NULL);
    for (unsigned int i = 0; i < data_fc->biases->size[0]; i++) {
        QVERIFY(data_fc->biases->data[i] == 0.0);
    }

    pearl_layer *output = pearl_layer_create_output(1);
    pearl_layer_add_child(&fc, &output);
    QCOMPARE(fc->num_child_layers, 1);
    QVERIFY(fc->child_layers != NULL);
    QVERIFY(fc->child_layers[0] != NULL);
    QVERIFY(fc->child_layers[0] == output);
    QCOMPARE(output->type, pearl_layer_type_output);
    QCOMPARE(output->num_child_layers, 0);
    QVERIFY(output->child_layers == NULL);
    QCOMPARE(output->num_parent_layers, 1);
    QVERIFY(output->parent_layers != NULL);
    QVERIFY(output->parent_layers[0] != NULL);
    QCOMPARE(output->parent_layers[0]->type, pearl_layer_type_fully_connected);
    QVERIFY(output->layer_data != NULL);
    QVERIFY(output->layer_data != NULL);
    pearl_layer_data_input *data_output = (pearl_layer_data_input *)output->layer_data;
    QCOMPARE(data_output->num_neurons, 1);
    QCOMPARE(data_output->activation_function, pearl_activation_function_type_linear);

    pearl_network_destroy(&network);
    QVERIFY(network == NULL);
}

void PearlLibTestNetwork::testCaseSaveLoad()
{
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
}

void PearlLibTestNetwork::testCaseNetworkForwardCheck()
{
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
}

void PearlLibTestNetwork::testCaseNetworkClassification()
{

}

void PearlLibTestNetwork::testCaseNetworkRegression()
{
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
}

QTEST_APPLESS_MAIN(PearlLibTestNetwork)

#include "tst_pearllib_network.moc"
