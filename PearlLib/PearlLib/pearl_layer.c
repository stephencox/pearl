#include <pearl_layer.h>

pearl_layer *pearl_layer_create()
{
    pearl_layer *layer = malloc(sizeof(pearl_layer));
    layer->child_layers = NULL;
    layer->num_child_layers = 0;
    layer->num_parent_layers = 0;
    layer->parent_layers = NULL;
    layer->layer_data = NULL;
    layer->a = NULL;
    layer->da = NULL;
    layer->dz = NULL;
    layer->z = NULL;
    return layer;
}

void pearl_layer_destroy(pearl_layer **layer)
{
    if (*layer != NULL) {
        if ((*layer)->parent_layers != NULL) {
            free((*layer)->parent_layers);
            (*layer)->parent_layers = NULL;
        }
        if ((*layer)->z != NULL) {
            pearl_tensor_destroy(&(*layer)->z);
        }
        if ((*layer)->a != NULL) {
            pearl_tensor_destroy(&(*layer)->a);
        }
        if ((*layer)->layer_data != NULL) {
            pearl_layer_data_fully_connected *data_fully_connected;
            pearl_layer_data_dropout *data_dropout;
            switch ((*layer)->type) {
                case pearl_layer_type_input:
                    break;
                case pearl_layer_type_fully_connected:
                    data_fully_connected = (pearl_layer_data_fully_connected *)(*layer)->layer_data;
                    if (data_fully_connected->biases != NULL) {
                        pearl_tensor_destroy(&data_fully_connected->biases);
                    }
                    if (data_fully_connected->weights != NULL) {
                        pearl_tensor_destroy(&data_fully_connected->weights);
                    }
                    break;
                case pearl_layer_type_dropout:
                    data_dropout = (*layer)->layer_data;
                    if (data_dropout->weights != NULL) {
                        pearl_tensor_destroy(&data_dropout->weights);
                    }
                    break;
            }
            free((*layer)->layer_data);
            (*layer)->layer_data = NULL;
        }
        if ((*layer)->child_layers != NULL) {
            for (unsigned int i = 0; i < (*layer)->num_child_layers; i++) {
                pearl_layer_destroy(&(*layer)->child_layers[i]);
            }
            free((*layer)->child_layers);
            (*layer)->child_layers = NULL;
        }
        free(*layer);
        *layer = NULL;
    }
}

PEARL_API void pearl_layer_add_child(pearl_layer **parent, pearl_layer **child)
{
    if (*parent != NULL) {
        if ((*parent)->child_layers == NULL) {
            (*parent)->num_child_layers = 1;
            (*parent)->child_layers = calloc(1, sizeof(pearl_layer *));
        }
        else {
            (*parent)->num_child_layers++;
            (*parent)->child_layers = realloc((*parent)->child_layers, (*parent)->num_child_layers * sizeof(pearl_layer *));
        }
        (*parent)->child_layers[(*parent)->num_child_layers - 1] = (*child);
        if ((*child)->parent_layers == NULL) {
            (*child)->num_parent_layers = 1;
            (*child)->parent_layers = calloc(1, sizeof(pearl_layer *));
        }
        else {
            (*child)->num_parent_layers++;
            (*child)->parent_layers = realloc((*child)->parent_layers, (*child)->num_child_layers * sizeof(pearl_layer *));
        }
        (*child)->parent_layers[(*child)->num_parent_layers - 1] = (*parent);
    }
}

pearl_layer *pearl_layer_create_input(unsigned int num_neurons)
{
    pearl_layer *layer = pearl_layer_create();
    layer->type = pearl_layer_type_input;
    layer->activation = pearl_activation_create(pearl_activation_function_type_linear);
    layer->num_neurons = num_neurons;
    return layer;
}

pearl_layer *pearl_layer_create_fully_connected(unsigned int num_neurons, unsigned int num_neurons_prev_layer)
{
    pearl_layer *layer = pearl_layer_create();
    layer->type = pearl_layer_type_fully_connected;
    layer->activation = pearl_activation_create(pearl_activation_function_type_relu);
    layer->num_neurons = num_neurons;
    pearl_layer_data_fully_connected *data = malloc(sizeof(pearl_layer_data_fully_connected));
    data->biases = pearl_tensor_create(1, layer->num_neurons);
    data->weights = pearl_tensor_create(2, layer->num_neurons, num_neurons_prev_layer);
    double var = sqrt(2.0 / (layer->num_neurons + num_neurons_prev_layer));
    for (unsigned int i = 0; i < data->weights->size[0] * data->weights->size[1]; i++) {
        data->weights->data[i] = pearl_util_rand_norm(0.0, var);
    }
    data->db = NULL;
    data->dw = NULL;
    layer->layer_data = data;
    return layer;
}

pearl_layer *pearl_layer_create_dropout(unsigned int num_neurons)
{
    pearl_layer *layer = pearl_layer_create();
    layer->type = pearl_layer_type_dropout;
    layer->num_neurons = num_neurons;
    pearl_layer_data_dropout *data = malloc(sizeof(pearl_layer_data_dropout));
    data->rate = 0.5;
    data->weights = pearl_tensor_create(1, layer->num_neurons);
    layer->layer_data = data;
    return layer;
}

void pearl_layer_forward(pearl_layer **parent_layer, pearl_layer **child_layer)
{
    switch ((*child_layer)->type) {
        case pearl_layer_type_input:
            break;
        case pearl_layer_type_fully_connected:
            pearl_layer_forward_fully_connected(parent_layer, child_layer);
            break;
        case pearl_layer_type_dropout:
            break;
    }
    for (unsigned int i = 0; i < (*child_layer)->num_child_layers; i++) {
        pearl_layer_forward(child_layer, &(*child_layer)->child_layers[i]);
    }
}

void pearl_layer_forward_fully_connected(pearl_layer **parent_layer, pearl_layer **child_layer)
{
    pearl_tensor *input = (*parent_layer)->a;
    pearl_layer_data_fully_connected *data = (pearl_layer_data_fully_connected *)(*child_layer)->layer_data;
    if ((*child_layer)->z == NULL) {
        (*child_layer)->z = pearl_tensor_create(2, data->weights->size[0], input->size[1]);
    }
    if ((*child_layer)->a == NULL) {
        (*child_layer)->a = pearl_tensor_create(2, data->weights->size[0], input->size[1]);
    }
    assert(data->weights->size[1] == input->size[0]);
    assert(data->weights->dimension == 2);
    assert(data->biases->dimension == 1);
    assert(data->weights->size[0] == data->biases->size[0]);
    for (unsigned int i = 0; i < data->weights->size[0]; i++) {
        for (unsigned int j = 0; j < input->size[1]; j++) {
            double sum = 0.0;
            for (unsigned int k = 0; k < data->weights->size[1]; k++) {
                assert(ARRAY_IDX_2D(i, k, data->weights->size[1]) < data->weights->size[0] * data->weights->size[1]);
                assert(ARRAY_IDX_2D(k, j, input->size[1]) < input->size[0]*input->size[1]);
                sum += data->weights->data[ARRAY_IDX_2D(i, k, data->weights->size[1])] * input->data[ARRAY_IDX_2D(k, j, input->size[1])];
            }
            sum += data->biases->data[i];
            assert(ARRAY_IDX_2D(i, j, (*child_layer)->z->size[1]) < (*child_layer)->z->size[0] * (*child_layer)->z->size[1]);
            (*child_layer)->z->data[ARRAY_IDX_2D(i, j, (*child_layer)->z->size[1])] = sum;
            assert(ARRAY_IDX_2D(i, j, (*child_layer)->a->size[1]) < (*child_layer)->a->size[0] * (*child_layer)->a->size[1]);
            (*child_layer)->a->data[ARRAY_IDX_2D(i, j, (*child_layer)->a->size[1])] = (*child_layer)->activation.calculate(sum);
        }
    }
}

void pearl_layer_backward(pearl_layer **child_layer, pearl_layer **parent_layer)
{
    switch ((*child_layer)->type) {
        case pearl_layer_type_input:
            break;
        case pearl_layer_type_fully_connected:
            pearl_layer_backward_fully_connected(child_layer, parent_layer);
            break;
        case pearl_layer_type_dropout:
            break;
    }
    for (unsigned int i = 0; i < (*parent_layer)->num_parent_layers; i++) {
        pearl_layer_backward(parent_layer, &(*parent_layer)->parent_layers[i]);
    }


}

void pearl_layer_backward_fully_connected(pearl_layer **child_layer, pearl_layer **parent_layer)
{
    /* Calculate dZ backward with activation derivative */
    assert((*child_layer)->layer_data != NULL);
    pearl_layer_data_fully_connected *data = (pearl_layer_data_fully_connected *)(*child_layer)->layer_data;
    assert((*child_layer)->da != NULL);
    assert((*parent_layer)->a != NULL);
    if ((*child_layer)->dz == NULL) {
        (*child_layer)->dz = pearl_tensor_create(2, (*child_layer)->da->size[0], (*child_layer)->da->size[1]);
    }
    if ((*parent_layer)->da == NULL) {
        (*parent_layer)->da = pearl_tensor_create(2, (*parent_layer)->a->size[0], (*child_layer)->dz->size[1]);
    }
    for (unsigned int j = 0; j < (*child_layer)->dz->size[1]; j++) {
        for (unsigned int x = 0; x < (*child_layer)->dz->size[0]; x++) {
            assert(ARRAY_IDX_2D(j, x, (*child_layer)->z->size[0]) < (*child_layer)->z->size[0] * (*child_layer)->z->size[1]);
            assert(ARRAY_IDX_2D(x, j, (*child_layer)->dz->size[1]) < (*child_layer)->dz->size[0] * (*child_layer)->dz->size[1]);
            assert(ARRAY_IDX_2D(x, j, (*child_layer)->da->size[1]) < (*child_layer)->da->size[0] * (*child_layer)->da->size[1]);
            (*child_layer)->dz->data[ARRAY_IDX_2D(x, j, (*child_layer)->dz->size[1])] = (*child_layer)->activation.calculate_derivative((*child_layer)->z->data[ARRAY_IDX_2D(x, j, (*child_layer)->z->size[1])]) * (*child_layer)->da->data[ARRAY_IDX_2D(x, j, (*child_layer)->da->size[1])];
        }
    }

    /* Calculate dW and dB of child layer */
    assert((*child_layer)->dz->dimension == 2);
    assert((*child_layer)->dz->dimension == 2);
    assert((*parent_layer)->a->dimension == 2);
    assert((*child_layer)->dz->size[1] == (*parent_layer)->a->size[1]);
    if (data->dw == NULL) {
        data->dw = pearl_tensor_create(2, (*child_layer)->da->size[0], (*parent_layer)->a->size[0]);
    }

    if (data->db == NULL) {
        data->db = pearl_tensor_create(1, (*child_layer)->da->size[0]);
    }
    assert(data->dw->dimension == 2);
    assert(data->dw->size[0] == (*child_layer)->dz->size[0]);
    assert(data->dw->size[1] == (*parent_layer)->a->size[0]);
    assert(data->db->dimension == 1);
    for (unsigned int i = 0; i < (*child_layer)->dz->size[0]; i++) {
        for (unsigned int j = 0; j < (*parent_layer)->a->size[0]; j++) {
            double sum_w = 0.0;
            double sum_b = 0.0;
            for (unsigned int k = 0; k < (*child_layer)->dz->size[1]; k++) {
                assert(ARRAY_IDX_2D(i, k, (*child_layer)->dz->size[1]) < (*child_layer)->dz->size[0] * (*child_layer)->dz->size[1]);
                assert(ARRAY_IDX_2D(j, k, (*parent_layer)->a->size[1]) < (*parent_layer)->a->size[0] * (*parent_layer)->a->size[1]);
                sum_w += (*child_layer)->dz->data[ARRAY_IDX_2D(i, k, (*child_layer)->dz->size[1])] * (*parent_layer)->a->data[ARRAY_IDX_2D(j, k, (*parent_layer)->a->size[1])];
                sum_b += (*child_layer)->dz->data[ARRAY_IDX_2D(i, k, (*child_layer)->dz->size[1])]; //TODO: remove duplicate add
            }
            assert(ARRAY_IDX_2D(i, j, data->dw->size[1]) < data->dw->size[0]*data->dw->size[1]);
            data->dw->data[ARRAY_IDX_2D(i, j, data->dw->size[1])] = sum_w / (*parent_layer)->a->size[1];
            assert(i < data->db->size[0]);
            data->db->data[i] = sum_b / (*parent_layer)->a->size[1];
        }
    }
    /* Calculate dA of parent layer */
    for (unsigned int i = 0; i < data->weights->size[1]; i++) {
        for (unsigned int j = 0; j < (*child_layer)->dz->size[1]; j++) {
            double sum_w = 0.0;
            for (unsigned int k = 0; k < data->weights->size[0]; k++) {
                assert(ARRAY_IDX_2D(k, i, data->weights->size[1]) < data->weights->size[0] * data->weights->size[1]);
                assert(ARRAY_IDX_2D(k, j, (*child_layer)->dz->size[1]) < (*child_layer)->dz->size[0] * (*child_layer)->dz->size[1]);
                sum_w += data->weights->data[ARRAY_IDX_2D(k, i, data->weights->size[1])] * (*child_layer)->dz->data[ARRAY_IDX_2D(k, j, (*child_layer)->da->size[1])];
            }
            assert(ARRAY_IDX_2D(i, j, (*parent_layer)->da->size[1]) < (*parent_layer)->da->size[0] * (*parent_layer)->da->size[1]);
            (*parent_layer)->da->data[ARRAY_IDX_2D(i, j, (*parent_layer)->da->size[1])] = sum_w;
        }
    }
}


void pearl_layer_update(pearl_layer **child_layer, double learning_rate)
{
    switch ((*child_layer)->type) {
        case pearl_layer_type_input:
            break;
        case pearl_layer_type_fully_connected:
            pearl_layer_update_fully_connected(child_layer, learning_rate);
            break;
        case pearl_layer_type_dropout:
            break;
    }
    for (unsigned int i = 0; i < (*child_layer)->num_child_layers; i++) {
        pearl_layer_update(&(*child_layer)->child_layers[i], learning_rate);
    }
}

void pearl_layer_update_fully_connected(pearl_layer **child_layer, double learning_rate)
{
    pearl_layer_data_fully_connected *data = (pearl_layer_data_fully_connected *)(*child_layer)->layer_data;
    assert(data->weights->dimension == 2);
    assert(data->weights->size[0] == data->dw->size[0]);
    assert(data->weights->size[1] == data->dw->size[1]);
    for (unsigned int i = 0; i < data->weights->size[0]; i++) {
        for (unsigned int j = 0; j < data->weights->size[1]; j++) {
            data->weights->data[ARRAY_IDX_2D(i, j, data->weights->size[1])] -= learning_rate * data->dw->data[ARRAY_IDX_2D(i, j, data->dw->size[1])];
        }
    }
    for (unsigned int i = 0; i < data->biases->size[0]; i++) {
        data->biases->data[i] -= learning_rate * data->db->data[i];
    }
}
