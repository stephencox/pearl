#include <pearl_network.h>

// TODO: Save network in sqlite

PEARL_API pearl_network *pearl_network_create()
{
    pearl_network *network = malloc(sizeof(pearl_network));
    network->num_layers = 0;
    network->optimiser = pearl_optimiser_sgd;
    network->learning_rate = 1e-3;
    return network;
}

PEARL_API void pearl_network_destroy(pearl_network *network)
{
    if (network) {
        if (network->layers) {
            for (int i = 0; i < network->num_layers; i++) {
                pearl_layer_destroy(&network->layers[i]);
            }
            free(network->layers);
        }
        free(network);
    }
}

PEARL_API void pearl_network_layer_add(pearl_network *network, enum pearl_layer_type type, int neurons, enum pearl_activation_function_type activation_function)
{
    network->num_layers++;
    if (network->num_layers > 1) {
        network->layers = (pearl_layer *)realloc(network->layers, network->num_layers * sizeof(pearl_layer)); //TODO: error checking
    }
    else {
        network->layers = (pearl_layer *)malloc(sizeof(pearl_layer));
    }
    network->layers[network->num_layers - 1].type = type;
    network->layers[network->num_layers - 1].neurons = neurons;
    network->layers[network->num_layers - 1].activation_function = activation_function;
    network->layers[network->num_layers - 1].weights = NULL;
    network->layers[network->num_layers - 1].biases = NULL;
}

PEARL_API void pearl_network_layer_add_input(pearl_network *network, int neurons)
{
    pearl_network_layer_add(network, pearl_layer_type_input, neurons, pearl_activation_function_type_linear);
}

PEARL_API void pearl_network_layer_add_output(pearl_network *network, int neurons, enum pearl_activation_function_type activation_function)
{
    pearl_network_layer_add(network, pearl_layer_type_output, neurons, activation_function);
}

PEARL_API void pearl_network_layer_add_dropout(pearl_network *network, int neurons, enum pearl_activation_function_type activation_function, double dropout_rate)
{
    pearl_network_layer_add(network, pearl_layer_type_dropout, neurons, activation_function);
    network->layers[network->num_layers - 1].dropout_rate = dropout_rate;
}

PEARL_API void pearl_network_layer_add_fully_connect(pearl_network *network, int neurons, enum pearl_activation_function_type activation_function)
{
    pearl_network_layer_add(network, pearl_layer_type_fully_connect, neurons, activation_function);
}

PEARL_API void pearl_network_layers_initialise(pearl_network *network)
{
    srand((unsigned int)time(NULL));
    for (int i = 1; i < network->num_layers; i++) {
        pearl_layer_initialise(&network->layers[i], &network->layers[i - 1]);
    }
}

PEARL_API void pearl_network_train_epoch(pearl_network *network, const pearl_matrix *input, const pearl_matrix *output)
{
    pearl_matrix **z = calloc(network->num_layers - 1, sizeof(pearl_matrix *));
    pearl_matrix **a = calloc(network->num_layers - 1, sizeof(pearl_matrix *));
    // Forward
    a[0] = pearl_matrix_copy(input);
    for (int i = 1; i < network->num_layers-1; i++) {
        assert(z[i] == NULL);
        z[i] = pearl_matrix_create(input->m, network->layers[i].weights->m);
        printf("malloced matrix in z[%d] at %p\n", i, (void *)z[i]);
        assert(a[i] == NULL);
        a[i] = pearl_matrix_create(input->m, network->layers[i].weights->m);
        printf("malloced matrix in a[%d] at %p\n", i, (void *)a[i]);
        pearl_layer_forward(&network->layers[i], a[i-1], z[i], a[i]);
        printf("Result:");
        pearl_matrix_print(a[i]);
        pearl_matrix_print(z[i]);
    }
    // Cost
    double cost = 0.0;
    pearl_matrix *al = a[network->num_layers - 2];
    for (int i = 0; i < output->m; i++) {
        if (output->data[i] > 0.0) {
            cost += log(al->data[i]);
        }
        else {
            cost += log(1.0 - al->data[i]);
        }
    }
    cost /= (double)(-output->n);
    printf("Loss: %f\n", cost);

    //Backward
    pearl_matrix **dw = calloc(network->num_layers - 1, sizeof(pearl_matrix *));
    pearl_vector **db = calloc(network->num_layers - 1, sizeof(pearl_vector *));
    pearl_matrix **dz = calloc(network->num_layers - 1, sizeof(pearl_matrix *));
    for (int i = network->num_layers - 1; i > 1; i--) {
        if (i == network->num_layers - 1) {
            assert(dz[i - 1] == NULL);
            dz[i - 1] = pearl_matrix_create(output->m, output->n);
            for (int j = 0; j < output->m; j++) {
                if (output->data[j] > 0.0) {
                    dz[i - 1]->data[j] = - (output->data[j] / al->data[j]);
                }
                else {
                    dz[i - 1]->data[j] = - ((1.0 - output->data[j]) / (1.0 - al->data[j]));
                }
            }
            assert(dw[i - 1] == NULL);
            dw[i - 1] = pearl_matrix_create(dz[i-1]->m, a[i-1]->m);
            assert(db[i - 1] == NULL);
            db[i - 1] = pearl_vector_create(a[i-1]->m);
            assert(dz[i - 2] == NULL);
            printf("Going to create dz[%d]\n", i-2);
            dz[i - 2] = pearl_layer_backward(&network->layers[i], &network->layers[i-1], dz[i - 1], a[i - 1], z[i - 1], dw[i - 1], db[i - 1]);
            printf("Created dz[%d] at %p\n", i-2, (void *)dz[i-2]);
        }
        else if (i == 1) {
            assert(dw[i - 1] == NULL);
            dw[i - 1] = pearl_matrix_create(input->m, network->layers[i].weights->m);
            assert(db[i - 1] == NULL);
            db[i - 1] = pearl_vector_create(input->m);
            assert(dz[i - 2] == NULL);
            dz[i - 2] = pearl_layer_backward(&network->layers[i], &network->layers[i-1], dz[i - 1], a[i - 1], z[i - 1], dw[i - 1], db[i - 1]);
        }
        else {
            assert(dw[i - 1] == NULL);
            dw[i - 1] = pearl_matrix_create(input->m, network->layers[i].weights->m);
            assert(db[i - 1] == NULL);
            db[i - 1] = pearl_vector_create(input->m);
            assert(dz[i - 2] == NULL);
            dz[i - 2] = pearl_layer_backward(&network->layers[i], &network->layers[i-1], dz[i - 1], a[i - 1], z[i - 1], dw[i - 1], db[i - 1]);
        }
        printf("Result:");
        pearl_matrix_print(z[i - 1]);
    }
    // Clean
    for (int i = 0; i < network->num_layers - 1; i++) {
        pearl_matrix_destroy(a[i]);
        pearl_matrix_destroy(z[i]);
        pearl_matrix_destroy(dw[i]);
        pearl_vector_destroy(db[i]);
        printf("Free dz[%d] at %p\n", i, (void *)dz[i]);
        pearl_matrix_destroy(dz[i]);
    }
    free(a);
    free(z);
    free(dw);
    free(db);
    free(dz);
}
