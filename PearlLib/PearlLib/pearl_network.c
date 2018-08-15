#include <pearl_network.h>

PEARL_API pearl_network *pearl_network_create()
{
    srand((unsigned int)time(NULL));
    pearl_network *network = malloc(sizeof(pearl_network));
    network->optimiser = pearl_optimiser_sgd;
    network->loss = pearl_loss_create(pearl_loss_binary_cross_entropy);
    network->learning_rate = 1e-3;
    network->version.major = PEARL_NETWORK_VERSION_MAJOR;
    network->version.minor = PEARL_NETWORK_VERSION_MINOR;
    network->version.revision = PEARL_NETWORK_VERSION_REVISION;
    return network;
}

PEARL_API void pearl_network_destroy(pearl_network **network)
{
    if (*network != NULL) {
        if ((*network)->input_layer != NULL) {
            pearl_layer_destroy(&(*network)->input_layer);
            free((*network)->input_layer);
            (*network)->input_layer = NULL;
        }
        free(*network);
        *network = NULL;
    }
}

PEARL_API double pearl_network_train_epoch(pearl_network **network, const pearl_tensor *input, const pearl_tensor *output)
{
    // Forward
    pearl_network_forward(network, input);

    // Cost
    pearl_tensor *al = (*network)->output_layer->a;
    double cost = pearl_loss_cost((*network)->loss, output, al);
    /*
        //Backward
        pearl_tensor **dw = calloc((*network)->num_layers, sizeof(pearl_tensor *));
        pearl_tensor **db = calloc((*network)->num_layers, sizeof(pearl_tensor *));
        pearl_tensor **dZ = calloc((*network)->num_layers, sizeof(pearl_tensor *));
        pearl_tensor **dA = calloc((*network)->num_layers + 1, sizeof(pearl_tensor *));
        unsigned int i;
        for (unsigned int num_layer = (*network)->num_layers; num_layer > 0; num_layer--) {
            i = num_layer - 1;
            if (i == (*network)->num_layers - 1) {
                assert(dA[i + 1] == NULL);
                dA[i + 1] = pearl_tensor_create(2, output->size[0], output->size[1]);
                for (unsigned int j = 0; j < output->size[1]; j++) {
                    for (unsigned int x = 0; x < output->size[0]; x++) {
                        assert(ARRAY_IDX_2D(j, x, output->size[0]) < output->size[0]*output->size[1]);
                        assert(ARRAY_IDX_2D(x, j, dA[i + 1]->size[1]) < dA[i + 1]->size[0]*dA[i + 1]->size[1]);
                        assert(ARRAY_IDX_2D(x, j, al->size[1]) < al->size[0]*al->size[1]);
                        dA[i + 1]->data[ARRAY_IDX_2D(x, j, dA[i + 1]->size[1])] = - (*network)->loss.calculate_derivative(output->data[ARRAY_IDX_2D(j, x, output->size[0])], al->data[ARRAY_IDX_2D(x, j, al->size[1])]);
                    }
                }
            }

            assert(dw[i] == NULL);
            assert(dA[i + 1] != NULL);
            dw[i] = pearl_tensor_create(2, dA[i + 1]->size[0], a[i]->size[0]);
            assert(db[i] == NULL);
            db[i] = pearl_tensor_create(1, dA[i + 1]->size[0]);
            assert(dZ[i] == NULL);
            dZ[i] = pearl_tensor_create(2, dA[i + 1]->size[0], dA[i + 1]->size[1]);
            assert(dA[i] == NULL);
            dA[i] = pearl_tensor_create(2, (*network)->layers[i]->weights->size[1], dZ[i]->size[1]);
            double (*activationFunctionDerivativePtr)(double) = pearl_activation_function_derivative_pointer((*network)->layers[i]->activation_function);

            for (unsigned int j = 0; j < dZ[i]->size[1]; j++) {
                for (unsigned int x = 0; x < dZ[i]->size[0]; x++) {
                    assert(ARRAY_IDX_2D(j, x, z[i]->size[0]) < z[i]->size[0]*z[i]->size[1]);
                    assert(ARRAY_IDX_2D(x, j, dZ[i]->size[1]) < dZ[i]->size[0]*dZ[i]->size[1]);
                    assert(ARRAY_IDX_2D(x, j, dA[i + 1]->size[1]) < dA[i + 1]->size[0]*dA[i + 1]->size[1]);
                    dZ[i]->data[ARRAY_IDX_2D(x, j, dZ[i]->size[1])] = activationFunctionDerivativePtr(z[i]->data[ARRAY_IDX_2D(x, j, z[i]->size[1])]) * dA[i + 1]->data[ARRAY_IDX_2D(x, j, dA[i + 1]->size[1])];
                }
            }
            pearl_layer_backward((*network)->layers[i], dZ[i], a[i], &dw[i], &db[i], &dA[i]);
        }
        //Update
        for (unsigned int i = 0; i < (*network)->num_layers; i++) {
            pearl_layer_update((*network)->layers[i], dw[i], db[i], (*network)->learning_rate);
        }
        // Clean
        for (unsigned int i = 0; i < (*network)->num_layers; i++) {
            pearl_tensor_destroy(&a[i]);
            pearl_tensor_destroy(&z[i]);
            pearl_tensor_destroy(&dw[i]);
            pearl_tensor_destroy(&db[i]);
            pearl_tensor_destroy(&dA[i]);
            pearl_tensor_destroy(&dZ[i]);
        }
        pearl_tensor_destroy(&a[(*network)->num_layers]);
        pearl_tensor_destroy(&dA[(*network)->num_layers]);
        free(a);
        free(z);
        free(dw);
        free(db);
        free(dA);
        free(dZ);
    */
    return cost;

}

void pearl_network_forward(pearl_network **network, const pearl_tensor *input)
{
    /* Initialise input layer */
    if ((*network)->input_layer->a == NULL) {
        (*network)->input_layer->z = pearl_tensor_create(2, input->size[1], input->size[0]);
        (*network)->input_layer->a = pearl_tensor_create(2, input->size[1], input->size[0]);
        for (unsigned int i = 0; i < input->size[0]; i++) {
            for (unsigned int j = 0; j < input->size[1]; j++) {
                (*network)->input_layer->a->data[ARRAY_IDX_2D(j, i, (*network)->input_layer->a->size[1])] = input->data[ARRAY_IDX_2D(i, j, input->size[1])];
                (*network)->input_layer->z->data[ARRAY_IDX_2D(j, i, (*network)->input_layer->z->size[1])] = (*network)->input_layer->activation.calculate(input->data[ARRAY_IDX_2D(i, j, input->size[1])]);
            }
        }
    }
    for (unsigned int i = 0; i < (*network)->input_layer->num_child_layers; i++) {
        pearl_layer_forward(&(*network)->input_layer, &(*network)->input_layer->child_layers[i]);
    }
}

PEARL_API pearl_tensor *pearl_network_calculate(pearl_network **network, const pearl_tensor *input)
{
    /*pearl_tensor **z = calloc((*network)->num_layers, sizeof(pearl_tensor *));
    pearl_tensor **a = calloc((*network)->num_layers + 1, sizeof(pearl_tensor *));
    pearl_network_forward(network, input, z, a);
    pearl_tensor *output = pearl_tensor_copy(a[(*network)->num_layers]);
    for (unsigned int i = 0; i < (*network)->num_layers; i++) {
        pearl_tensor_destroy(&a[i]);
        pearl_tensor_destroy(&z[i]);
    }
    pearl_tensor_destroy(&a[(*network)->num_layers]);
    free(a);
    free(z);
    return output;*/
}
