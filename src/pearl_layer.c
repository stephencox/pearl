#include <pearl_layer.h>
#include <omp.h>

pearl_layer *pearl_layer_create()
{
    pearl_layer *layer = calloc(1, sizeof(pearl_layer));
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
        // Free parent layers array
        if ((*layer)->parent_layers != NULL) {
            free((*layer)->parent_layers);
        }

        // Destroy z tensor
        if ((*layer)->z != NULL) {
            pearl_tensor_destroy(&(*layer)->z);
        }

        if ((*layer)->dz != NULL) {
            pearl_tensor_destroy(&(*layer)->dz);
        }

        // Destroy a tensor
        if ((*layer)->a != NULL) {
            pearl_tensor_destroy(&(*layer)->a);
        }

        if ((*layer)->da != NULL) {
            pearl_tensor_destroy(&(*layer)->da);
        }

        // Handle layer-specific data
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

                    if (data_fully_connected->db != NULL) {
                        pearl_tensor_destroy(&data_fully_connected->db);
                    }

                    if (data_fully_connected->dw != NULL) {
                        pearl_tensor_destroy(&data_fully_connected->dw);
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
        }

        // Destroy child layers
        if ((*layer)->child_layers != NULL) {
            for (unsigned int i = 0; i < (*layer)->num_child_layers; i++) {
                pearl_layer_destroy(&(*layer)->child_layers[i]);
            }
            free((*layer)->child_layers);
        }

        // Free the layer itself
        free(*layer);
        *layer = NULL;
    }
}

PEARL_API void pearl_layer_add_child(pearl_layer **parent, pearl_layer **child)
{
    if (*parent != NULL) {
        if ((*parent)->child_layers == NULL) {
            (*parent)->child_layers = calloc(1, sizeof(pearl_layer *));
        }
        else {
            (*parent)->child_layers = realloc((*parent)->child_layers, ((*parent)->num_child_layers + 1) * sizeof(pearl_layer *));
        }
        if((*parent)->child_layers == NULL){
            return;
        }
        (*parent)->child_layers[(*parent)->num_child_layers] = (*child);
        (*parent)->num_child_layers++;

        if ((*child)->parent_layers == NULL) {
            (*child)->parent_layers = calloc(1, sizeof(pearl_layer *));
        }
        else {
            (*child)->parent_layers = realloc((*child)->parent_layers, ((*child)->num_parent_layers + 1) * sizeof(pearl_layer *));
        }
        if((*child)->parent_layers == NULL){
            return;
        }
        (*child)->parent_layers[(*child)->num_parent_layers] = (*parent);
        (*child)->num_parent_layers++;
    }
}

pearl_layer *pearl_layer_create_input(unsigned int num_neurons)
{
    pearl_layer *layer = pearl_layer_create();
    layer->type = pearl_layer_type_input;
    layer->activation = pearl_activation_create(pearl_activation_type_linear);
    layer->num_neurons = num_neurons;
    return layer;
}

pearl_layer *pearl_layer_create_fully_connected(unsigned int num_neurons, unsigned int num_neurons_prev_layer)
{
    pearl_layer *layer = pearl_layer_create();
    layer->type = pearl_layer_type_fully_connected;
    layer->activation = pearl_activation_create(pearl_activation_type_relu);
    layer->num_neurons = num_neurons;
    pearl_layer_data_fully_connected *data = calloc(1, sizeof(pearl_layer_data_fully_connected));
    data->biases = pearl_tensor_create(1, layer->num_neurons);
    data->weights = pearl_tensor_create(2, layer->num_neurons, num_neurons_prev_layer);
    float var = sqrtf(2.0f / (float)(layer->num_neurons + num_neurons_prev_layer));
    for (unsigned int i = 0; i < data->weights->size[0] * data->weights->size[1]; i++) {
        data->weights->data[i] = pearl_util_rand_norm(0.0f, var);
    }
    data->db = NULL;
    data->dw = NULL;
    layer->layer_data = data;
    return layer;
}

pearl_layer *pearl_layer_create_fully_connected_blank(unsigned int num_neurons)
{
    pearl_layer *layer = pearl_layer_create();
    layer->type = pearl_layer_type_fully_connected;
    layer->activation = pearl_activation_create(pearl_activation_type_relu);
    layer->num_neurons = num_neurons;
    pearl_layer_data_fully_connected *data = calloc(1, sizeof(pearl_layer_data_fully_connected));
    data->biases = NULL;
    data->weights = NULL;
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
    pearl_layer_data_dropout *data = calloc(1, sizeof(pearl_layer_data_dropout));
    data->rate = 0.5f;
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
    const pearl_tensor *input = (*parent_layer)->a;
    const pearl_layer_data_fully_connected *data = (pearl_layer_data_fully_connected *)(*child_layer)->layer_data;
    pearl_tensor *z = (*child_layer)->z;
    pearl_tensor *a = (*child_layer)->a;

    if (z == NULL) {
        z = pearl_tensor_create(2, data->weights->size[0], input->size[1]);
        (*child_layer)->z = z;
    }
    if (a == NULL) {
        a = pearl_tensor_create(2, data->weights->size[0], input->size[1]);
        (*child_layer)->a = a;
    }

    assert(data->weights->size[1] == input->size[0]);
    assert(data->weights->dimension == 2);
    assert(data->biases->dimension == 1);
    assert(data->weights->size[0] == data->biases->size[0]);

    const unsigned int weights_size_0 = data->weights->size[0];
    const unsigned int input_size_1 = input->size[1];
    const unsigned int weights_size_1 = data->weights->size[1];
    const float *weights_data = data->weights->data;
    const float *input_data = input->data;
    const float *biases_data = data->biases->data;
    float *z_data = z->data;
    float *a_data = a->data;

    #pragma omp parallel for collapse(2)
    for (unsigned int i = 0; i < weights_size_0; i++) {
        for (unsigned int j = 0; j < input_size_1; j++) {
            float sum = 0.0f;
            for (unsigned int k = 0; k < weights_size_1; k++) {
                sum += weights_data[ARRAY_IDX_2D(i, k, weights_size_1)] * input_data[ARRAY_IDX_2D(k, j, input_size_1)];
            }
            sum += biases_data[i];
            z_data[ARRAY_IDX_2D(i, j, z->size[1])] = sum;
            a_data[ARRAY_IDX_2D(i, j, a->size[1])] = (*child_layer)->activation.calculate(sum);
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

    unsigned int dz_size_0 = (*child_layer)->dz->size[0];
    unsigned int dz_size_1 = (*child_layer)->dz->size[1];
    unsigned int a_size_0 = (*parent_layer)->a->size[0];
    unsigned int a_size_1 = (*parent_layer)->a->size[1];
    float a_size_1_float = (float)a_size_1;
    unsigned int da_size_1 = (*child_layer)->da->size[1];
    float *dz_data = (*child_layer)->dz->data;
    float *z_data = (*child_layer)->z->data;
    float *da_data = (*child_layer)->da->data;
    float *a_data = (*parent_layer)->a->data;
    float *dw_data = data->dw->data;
    float *db_data = data->db->data;
    float *weights_data = data->weights->data;
    unsigned int weights_size_1 = data->weights->size[1];

    #pragma omp parallel for
    for (unsigned int i = 0; i < dz_size_0; i++) {
        float sum_b = 0.0f;
        for (unsigned int k = 0; k < dz_size_1; k++) {
            dz_data[ARRAY_IDX_2D(i, k, dz_size_1)] = (*child_layer)->activation.calculate_derivative(z_data[ARRAY_IDX_2D(i, k, dz_size_1)]) * da_data[ARRAY_IDX_2D(i, k, da_size_1)];
            sum_b += dz_data[ARRAY_IDX_2D(i, k, dz_size_1)];
        }

        for (unsigned int j = 0; j < a_size_0; j++) {
            float sum_w = 0.0f;
            for (unsigned int k = 0; k < dz_size_1; k++) {
                assert(ARRAY_IDX_2D(i, k, dz_size_1) < dz_size_0 * dz_size_1);
                assert(ARRAY_IDX_2D(j, k, a_size_1) < a_size_0 * a_size_1);
                sum_w += dz_data[ARRAY_IDX_2D(i, k, dz_size_1)] * a_data[ARRAY_IDX_2D(j, k, a_size_1)];
            }
            assert(ARRAY_IDX_2D(i, j, data->dw->size[1]) < data->dw->size[0]*data->dw->size[1]);
            dw_data[ARRAY_IDX_2D(i, j, data->dw->size[1])] = sum_w / a_size_1_float;
        }
        assert(i < data->db->size[0]);
        db_data[i] = sum_b / a_size_1_float;
    }

    /* Calculate dA of parent layer */
    #pragma omp parallel for collapse(2)
    for (unsigned int i = 0; i < weights_size_1; i++) {
        for (unsigned int j = 0; j < dz_size_1; j++) {
            float sum_w = 0.0f;
            for (unsigned int k = 0; k < (*child_layer)->dz->size[0]; k++) {
                assert(ARRAY_IDX_2D(k, i, weights_size_1) < data->weights->size[0] * weights_size_1);
                assert(ARRAY_IDX_2D(k, j, dz_size_1) < (*child_layer)->dz->size[0] * dz_size_1);
                sum_w += weights_data[ARRAY_IDX_2D(k, i, weights_size_1)] * dz_data[ARRAY_IDX_2D(k, j, da_size_1)];
            }
            assert(ARRAY_IDX_2D(i, j, (*parent_layer)->da->size[1]) < (*parent_layer)->da->size[0] * (*parent_layer)->da->size[1]);
            (*parent_layer)->da->data[ARRAY_IDX_2D(i, j, (*parent_layer)->da->size[1])] = sum_w;
        }
    }
}

void pearl_layer_update(pearl_layer **child_layer, float learning_rate)
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

void pearl_layer_update_fully_connected(pearl_layer **child_layer, float learning_rate)
{
    pearl_layer_data_fully_connected *data = (pearl_layer_data_fully_connected *)(*child_layer)->layer_data;
    assert(data->weights->dimension == 2);
    assert(data->weights->size[0] == data->dw->size[0]);
    assert(data->weights->size[1] == data->dw->size[1]);
    for (unsigned int i = 0; i < data->weights->size[0]; i++) {
        for (unsigned int j = 0; j < data->weights->size[1]; j++) {
            data->weights->data[ARRAY_IDX_2D(i, j, data->weights->size[1])] -= learning_rate * data->dw->data[ARRAY_IDX_2D(i, j, data->dw->size[1])];
        }
        data->biases->data[i] -= learning_rate * data->db->data[i];
    }
}
