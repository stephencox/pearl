#include <pearl_layer.h>

pearl_layer *pearl_layer_create()
{
    pearl_layer *layer = malloc(sizeof(pearl_layer));
    layer->child_layers = NULL;
    layer->num_child_layers = 0;
    layer->num_parent_layers = 0;
    layer->parent_layers = NULL;
    layer->layer_data = NULL;
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

void pearl_layer_backward(const pearl_layer *layer, const pearl_tensor *dA, const pearl_tensor *a, pearl_tensor **dw, pearl_tensor **db, pearl_tensor **da_prev)
{
    /*pearl_tensor *dw_p = (*dw);
    pearl_tensor *db_p = (*db);
    pearl_tensor *da_prev_p = (*da_prev);
    assert(dA->dimension == 2);
    assert(dA->dimension == 2);
    assert(a->dimension == 2);
    assert(dA->size[1] == a->size[1]);
    assert(dw_p->dimension == 2);
    assert(dw_p->size[0] == dA->size[0]);
    assert(dw_p->size[1] == a->size[0]);
    assert(db_p->dimension == 1);
    for (unsigned int i = 0; i < dA->size[0]; i++) {
        for (unsigned int j = 0; j < a->size[0]; j++) {
            double sum_w = 0.0;
            double sum_b = 0.0;
            for (unsigned int k = 0; k < dA->size[1]; k++) {
                assert(ARRAY_IDX_2D(i, k, dA->size[1]) < dA->size[0] * dA->size[1]);
                assert(ARRAY_IDX_2D(j, k, a->size[1]) < a->size[0]*a->size[1]);
                sum_w += dA->data[ARRAY_IDX_2D(i, k, dA->size[1])] * a->data[ARRAY_IDX_2D(j, k, a->size[1])];
                sum_b += dA->data[ARRAY_IDX_2D(i, k, dA->size[1])]; //TODO: remove duplicate add
            }
            assert(ARRAY_IDX_2D(i, j, dw_p->size[1]) < dw_p->size[0]*dw_p->size[1]);
            dw_p->data[ARRAY_IDX_2D(i, j, dw_p->size[1])] = sum_w / a->size[1];
            assert(i < db_p->size[0]);
            db_p->data[i] = sum_b / a->size[1];
        }
    }
    for (unsigned int i = 0; i < layer->weights->size[1]; i++) {
        for (unsigned int j = 0; j < dA->size[1]; j++) {
            double sum_w = 0.0;
            for (unsigned int k = 0; k < layer->weights->size[0]; k++) {
                assert(ARRAY_IDX_2D(k, i, layer->weights->size[1]) < layer->weights->size[0] * layer->weights->size[1]);
                assert(ARRAY_IDX_2D(k, j, dA->size[1]) < dA->size[0]*dA->size[1]);
                sum_w += layer->weights->data[ARRAY_IDX_2D(k, i, layer->weights->size[1])] * dA->data[ARRAY_IDX_2D(k, j, dA->size[1])];
            }
            assert(ARRAY_IDX_2D(i, j, da_prev_p->size[1]) < da_prev_p->size[0]*da_prev_p->size[1]);
            da_prev_p->data[ARRAY_IDX_2D(i, j, da_prev_p->size[1])] = sum_w;
        }
    }*/
}

void pearl_layer_update(pearl_layer *layer, pearl_tensor *dw, pearl_tensor *db, double learning_rate)
{
    /*assert(layer->weights->dimension == 2);
    assert(layer->weights->size[0] == dw->size[0]);
    assert(layer->weights->size[1] == dw->size[1]);
    for (unsigned int i = 0; i < layer->weights->size[0]; i++) {
        for (unsigned int j = 0; j < layer->weights->size[1]; j++) {
            layer->weights->data[ARRAY_IDX_2D(i, j, layer->weights->size[1])] -= learning_rate * dw->data[ARRAY_IDX_2D(i, j, dw->size[1])];
        }
    }
    for (unsigned int i = 0; i < layer->biases->size[0]; i++) {
        layer->biases->data[i] -= learning_rate * db->data[i];
    }*/
}
