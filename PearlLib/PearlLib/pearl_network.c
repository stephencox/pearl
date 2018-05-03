#include <pearl_network.h>

PEARL_API pearl_network *pearl_network_create(const unsigned int num_input, const unsigned int num_output)
{
    srand((unsigned int)time(NULL));
    pearl_network *network = malloc(sizeof(pearl_network));
    network->num_layers = 0;
    network->optimiser = pearl_optimiser_sgd;
    network->loss_type = pearl_loss_binary_cross_entropy;
    network->learning_rate = 1e-3;
    network->num_input = num_input;
    network->num_output = num_output;
    network->layers = NULL;
    network->version.major = PEARL_NETWORK_VERSION_MAJOR;
    network->version.minor = PEARL_NETWORK_VERSION_MINOR;
    network->version.revision = PEARL_NETWORK_VERSION_REVISION;
    return network;
}

PEARL_API void pearl_network_destroy(pearl_network **network)
{
    if (network != NULL) {
        if ((*network)->layers) {
            for (int i = 0; i < (*network)->num_layers; i++) {
                pearl_layer_destroy(&(*network)->layers[i]);
            }
            free((*network)->layers);
            (*network)->layers = NULL;
        }
        free(*network);
        *network = NULL;
    }
}

PEARL_API void pearl_network_save(const char *filename, const pearl_network *network)
{
    FILE *f = fopen(filename, "wb");
    if (f == NULL) {
        printf("Saved failed: Error opening file!\n");
        return;
    }
    fwrite(&network->version, sizeof(pearl_version), 1, f);
    fwrite(&network->num_input, sizeof(unsigned int), 1, f);
    fwrite(&network->num_output, sizeof(unsigned int), 1, f);
    fwrite(&network->num_layers, sizeof(unsigned int), 1, f);
    fwrite(&network->learning_rate, sizeof(double), 1, f);
    fwrite(&network->loss_type, sizeof(pearl_loss), 1, f);
    fwrite(&network->optimiser, sizeof(pearl_optimiser), 1, f);
    for (int i = 0; i < network->num_layers; i++) {
        fwrite(&network->layers[i]->version, sizeof(pearl_version), 1, f);
        fwrite(&network->layers[i]->activation_function, sizeof(pearl_activation_function_type), 1, f);
        fwrite(&network->layers[i]->dropout_rate, sizeof(double), 1, f);
        fwrite(&network->layers[i]->neurons, sizeof(unsigned int), 1, f);
        fwrite(&network->layers[i]->type, sizeof(pearl_layer), 1, f);
        pearl_tensor_save(network->layers[i]->weights, f);
        pearl_tensor_save(network->layers[i]->biases, f);
    }
    fclose(f);
}

PEARL_API void pearl_network_layer_add(pearl_network **network, const pearl_layer_type type, const int neurons, const pearl_activation_function_type activation_function)
{
    pearl_network *network_p = (*network);
    network_p->num_layers++;
    if (network_p->num_layers > 1) {
        network_p->layers = realloc(network_p->layers, network_p->num_layers * sizeof(pearl_layer*)); //TODO: error checking
    }
    else {
        network_p->layers = calloc(1, sizeof(pearl_layer*));
    }
    pearl_layer *layer = malloc(sizeof(pearl_layer));
    layer->type = type;
    layer->neurons = neurons;
    layer->activation_function = activation_function;
    layer->weights = NULL;
    layer->biases = NULL;
    layer->dropout_rate = 0.0;
    layer->version.major = PEARL_LAYER_VERSION_MAJOR;
    layer->version.minor = PEARL_LAYER_VERSION_MINOR;
    layer->version.revision = PEARL_LAYER_VERSION_REVISION;
    network_p->layers[network_p->num_layers - 1] = layer;
}

PEARL_API void pearl_network_layer_add_output(pearl_network **network, const pearl_activation_function_type activation_function)
{
    pearl_network_layer_add(network, pearl_layer_type_output, (*network)->num_output, activation_function);
}

PEARL_API void pearl_network_layer_add_dropout(pearl_network **network, const int neurons, const pearl_activation_function_type activation_function, const double dropout_rate)
{
    pearl_network_layer_add(network, pearl_layer_type_dropout, neurons, activation_function);
    pearl_network *network_p = (*network);
    network_p->layers[network_p->num_layers - 1]->dropout_rate = dropout_rate;
}

PEARL_API void pearl_network_layer_add_fully_connect(pearl_network **network, const int neurons, const pearl_activation_function_type activation_function)
{
    pearl_network_layer_add(network, pearl_layer_type_fully_connect, neurons, activation_function);
}

PEARL_API void pearl_network_layers_initialise(pearl_network **network)
{
    pearl_network *network_p = (*network);
    if (network_p->layers) {
        for (int i = 0; i < network_p->num_layers; i++) {
            int num_neurons_next_layer = (i < network_p->num_layers-1 ? network_p->layers[i + 1]->neurons : network_p->num_output);
            pearl_layer_initialise(network_p->layers[i], num_neurons_next_layer);
        }
    }
}

PEARL_API void pearl_network_train_epoch(pearl_network **network, const pearl_tensor *input, const pearl_tensor *output)
{
    pearl_network *network_p = (*network);
    pearl_tensor **z = calloc(network_p->num_layers, sizeof(pearl_tensor *));
    pearl_tensor **a = calloc(network_p->num_layers + 1, sizeof(pearl_tensor *));
    // Forward
    a[0] = pearl_tensor_create(2, input->size[1], input->size[0]);
    for (unsigned int i = 0; i < input->size[0]; i++) {
        for (unsigned int j = 0; j < input->size[1]; j++) {
            a[0]->data[ARRAY_IDX_2D(j, i, a[0]->size[1])] = input->data[ARRAY_IDX_2D(i, j, input->size[1])];
        }
    }

    for (int i = 0; i < network_p->num_layers; i++) {
        printf("Layer %d\n", i);
        assert(z[i] == NULL);
        z[i] = pearl_tensor_create(2, network_p->layers[i]->weights->size[0], a[i]->size[1]);
        assert(a[i + 1] == NULL);
        a[i + 1] = pearl_tensor_create(2, network_p->layers[i]->weights->size[0], a[i]->size[1]);
        pearl_layer_forward(network_p->layers[i], a[i], z[i], a[i + 1]);
    }
    // Cost
    pearl_tensor *al = a[network_p->num_layers];
    double cost = 0.0;
    switch (network_p->loss_type) {
    case pearl_loss_binary_cross_entropy:
        cost = pearl_loss_binary_cross_entropy_cost(output, al);
        break;
    default:
        printf("Invalid loss function\n");
        break;
    }
    printf("Loss: %f\n", cost);

    //Backward
    pearl_tensor **dw = calloc(network_p->num_layers, sizeof(pearl_tensor *));
    pearl_tensor **db = calloc(network_p->num_layers, sizeof(pearl_tensor *));
    pearl_tensor **dz = calloc(network_p->num_layers, sizeof(pearl_tensor *));
    for (int i = network_p->num_layers - 1; i >= 0; i--) {
        if (i == network_p->num_layers - 1) {
            assert(dz[i] == NULL);
            dz[i] = pearl_tensor_create(2, output->size[1], output->size[0]);
            for (unsigned int j = 0; j < output->size[1]; j++) {
                for (unsigned int x = 0; x < output->size[0]; x++) {
                    assert(ARRAY_IDX_2D(j, x, output->size[0]) < output->size[0]*output->size[1]);
                    assert(ARRAY_IDX_2D(j, x, dz[i]->size[1]) < dz[i]->size[0]*dz[i]->size[1]);
                    assert(ARRAY_IDX_2D(j, x, al->size[1]) < al->size[0]*al->size[1]);
                    double temp = output->data[ARRAY_IDX_2D(j, x, output->size[0])];
                    if (temp > 0.0) {
                        dz[i]->data[ARRAY_IDX_2D(j, x, dz[i]->size[1])] = - (output->data[ARRAY_IDX_2D(j, x, output->size[0])] / al->data[ARRAY_IDX_2D(j, x, al->size[1])]);
                    }
                    else {
                        dz[i]->data[ARRAY_IDX_2D(j, x, dz[i]->size[1])] = ((1.0 - output->data[ARRAY_IDX_2D(j, x, output->size[0])]) / (1.0 - al->data[ARRAY_IDX_2D(j, x, al->size[1])]));
                    }
                }
            }
            printf("dZ at output\n");
            pearl_tensor_print(dz[i]);
        }
        if (i > 0) {
            assert(dw[i] == NULL);
            dw[i] = pearl_tensor_create(2, dz[i]->size[0], a[i]->size[0]);
            assert(db[i] == NULL);
            db[i] = pearl_tensor_create(1, dz[i]->size[0]);
            assert(dz[i - 1] == NULL);
            dz[i - 1] = pearl_layer_backward(network_p->layers[i], network_p->layers[i - 1], dz[i], a[i], z[i], dw[i], db[i]);
            pearl_tensor_print(dz[i - 1]);
        }
        else {
            assert(dw[i] == NULL);
            dw[i] = pearl_tensor_create(2, dz[i]->size[0], a[i]->size[0]);
            assert(db[i] == NULL);
            db[i] = pearl_tensor_create(1, dz[i]->size[0]);
            pearl_layer_backward_weights_biases(dz[i], a[i], dw[i], db[i]);
        }
    }
    //Update
    for (int i = 0; i < network_p->num_layers; i++) {
        pearl_layer_update(network_p->layers[i], dw[i], db[i], network_p->learning_rate);
    }
    // Clean
    for (int i = 0; i < network_p->num_layers - 1; i++) {
        pearl_tensor_destroy(&a[i]);
        pearl_tensor_destroy(&z[i]);
        pearl_tensor_destroy(&dw[i]);
        pearl_tensor_destroy(&db[i]);
        pearl_tensor_destroy(&dz[i]);
    }
    free(a);
    free(z);
    free(dw);
    free(db);
    free(dz);
}
