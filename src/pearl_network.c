#include <pearl_network.h>

PEARL_API pearl_network *pearl_network_create()
{
    srand((unsigned int)time(NULL));
    pearl_network *network = calloc(1, sizeof(pearl_network));
    network->optimiser = pearl_optimiser_sgd;
    network->loss = pearl_loss_create(pearl_loss_binary_cross_entropy);
    network->learning_rate = 1e-3f;
    network->version.major = PEARL_NETWORK_VERSION_MAJOR;
    network->version.minor = PEARL_NETWORK_VERSION_MINOR;
    network->version.revision = PEARL_NETWORK_VERSION_REVISION;
    network->input_layer = NULL;
    network->is_training = true; // Add this line
    return network;
}

PEARL_API void pearl_network_destroy(pearl_network **network)
{
    if (*network != NULL) {
        if ((*network)->input_layer != NULL) {
            pearl_layer_destroy(&((*network)->input_layer)); // This will free the layer and set (*network)->input_layer to NULL.
        }
        // No need to free (*network)->input_layer again or set it to NULL here.
        free(*network); // Free the network struct itself
        *network = NULL;
    }
}

PEARL_API float pearl_network_train_epoch(pearl_network **network, const pearl_tensor *input, const pearl_tensor *output)
{
    (*network)->is_training = true; // Set is_training to true for training
    // Forward
    pearl_network_forward(network, input);

    // Cost
    const pearl_tensor *al = (*network)->output_layer->a;
    float cost = pearl_loss_cost((*network)->loss, output, al);

    //Backward
    pearl_network_backward(network, output);

    //Update
    for (unsigned int i = 0; i < (*network)->input_layer->num_child_layers; i++) {
        pearl_layer_update(&(*network)->input_layer->child_layers[i], (*network)->learning_rate);
    }

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

    /* Recursive forward other layers */
    for (unsigned int i = 0; i < (*network)->input_layer->num_child_layers; i++) {
        pearl_layer_forward(&(*network)->input_layer, &(*network)->input_layer->child_layers[i], (*network)->is_training); // Pass is_training
    }
}

void pearl_network_backward(pearl_network **network, const pearl_tensor *output)
{
    /* Initialise loss derivative */
    if ((*network)->output_layer->da == NULL)
    {
        (*network)->output_layer->da = pearl_tensor_create(2, output->size[0], output->size[1]);
    }
    for (unsigned int j = 0; j < output->size[1]; j++) {
        for (unsigned int x = 0; x < output->size[0]; x++) {
            assert(ARRAY_IDX_2D(j, x, output->size[0]) < output->size[0]*output->size[1]);
            assert(ARRAY_IDX_2D(x, j, (*network)->output_layer->da->size[1]) < (*network)->output_layer->da->size[0] * (*network)->output_layer->da->size[1]);
            assert(ARRAY_IDX_2D(x, j, (*network)->output_layer->a->size[1]) < (*network)->output_layer->a->size[0] * (*network)->output_layer->a->size[1]);
            (*network)->output_layer->da->data[ARRAY_IDX_2D(x, j, (*network)->output_layer->da->size[1])] = - (*network)->loss.calculate_derivative(output->data[ARRAY_IDX_2D(j, x, output->size[0])], (*network)->output_layer->a->data[ARRAY_IDX_2D(x, j, (*network)->output_layer->a->size[1])]);
        }
    }

    /* Recursive forward other layers */
    for (unsigned int i = 0; i < (*network)->output_layer->num_parent_layers; i++) {
        pearl_layer_backward(&(*network)->output_layer, &(*network)->output_layer->parent_layers[i]);
    }
}

PEARL_API pearl_tensor *pearl_network_calculate(pearl_network **network, const pearl_tensor *input)
{
    (*network)->is_training = false; // Set is_training to false for calculation/inference
    pearl_network_forward(network, input);
    pearl_tensor *output = pearl_tensor_copy((*network)->output_layer->a);
    return output;
}
