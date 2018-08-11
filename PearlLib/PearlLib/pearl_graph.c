#include <pearl_graph.h>

pearl_graph *pearl_graph_create(unsigned int num_inputs)
{
    pearl_graph *graph = malloc(sizeof(pearl_graph));
    graph->num_layers = 0;
    graph->num_inputs = num_inputs;
    graph->inputs = calloc(num_inputs, sizeof(pearl_graph_layer *));
    return graph;
}

void pearl_graph_destroy(pearl_graph **graph)
{
    if (*graph != NULL) {
        if ((*graph)->inputs != NULL) {
            for (unsigned int i = 0; i < (*graph)->num_inputs; i++) {
                pearl_graph_layer_destroy(&(*graph)->inputs[i]);
            }
            free((*graph)->inputs);
            (*graph)->inputs = NULL;
        }
        free(*graph);
        *graph = NULL;
    }
}

PEARL_API void pearl_graph_add_child(pearl_graph_layer **parent, pearl_graph_layer **child)
{
    if (*parent != NULL) {
        if ((*parent)->childs == NULL) {
            (*parent)->num_childs = 1;
            (*parent)->childs = calloc(1, sizeof(pearl_graph_layer *));
        }
        else {
            (*parent)->num_childs++;
            (*parent)->childs = realloc((*parent)->childs, (*parent)->num_childs * sizeof(pearl_graph_layer *));
        }
        (*parent)->childs[(*parent)->num_childs - 1] = (*child);
        if ((*child)->parents == NULL) {
            (*child)->num_parents = 1;
            (*child)->parents = calloc(1, sizeof(pearl_graph_layer *));
        }
        else {
            (*child)->num_parents++;
            (*child)->parents = realloc((*child)->parents, (*child)->num_childs * sizeof(pearl_graph_layer *));
        }
        (*child)->parents[(*child)->num_parents - 1] = (*parent);
    }
}

pearl_graph_layer *pearl_graph_layer_create()
{
    pearl_graph_layer *layer = malloc(sizeof(pearl_graph_layer));
    layer->childs = NULL;
    layer->num_childs = 0;
    layer->num_parents = 0;
    layer->parents = NULL;
    layer->node_data = NULL;
    return layer;
}

void pearl_graph_layer_destroy(pearl_graph_layer **layer)
{
    if (*layer != NULL) {
        if ((*layer)->parents != NULL) {
            free((*layer)->parents);
            (*layer)->parents = NULL;
        }
        if ((*layer)->node_data != NULL) {
            pearl_graph_node_fully_connected *data_fully_connected;
            pearl_graph_node_dropout *data_dropout;
            switch ((*layer)->node_type) {
                case pearl_graph_layer_type_input:
                    break;
                case pearl_graph_layer_type_output:
                    break;
                case pearl_graph_layer_type_fully_connected:
                    data_fully_connected = (pearl_graph_node_fully_connected *)(*layer)->node_data;
                    if (data_fully_connected->biases != NULL) {
                        pearl_tensor_destroy(&data_fully_connected->biases);
                    }
                    if (data_fully_connected->weights != NULL) {
                        pearl_tensor_destroy(&data_fully_connected->weights);
                    }
                    break;
                case pearl_graph_layer_type_dropout:
                    data_dropout = (*layer)->node_data;
                    if (data_dropout->weights != NULL) {
                        pearl_tensor_destroy(&data_dropout->weights);
                    }
                    break;
            }
            free((*layer)->node_data);
            (*layer)->node_data = NULL;
        }
        if ((*layer)->childs != NULL) {
            for (unsigned int i = 0; i < (*layer)->num_childs; i++) {
                pearl_graph_layer_destroy(&(*layer)->childs[i]);
            }
            free((*layer)->childs);
            (*layer)->childs = NULL;
        }
        free(*layer);
        *layer = NULL;
    }
}

pearl_graph_layer *pearl_graph_layer_create_input(unsigned int num_neurons)
{
    pearl_graph_layer *layer = pearl_graph_layer_create();
    layer->node_type = pearl_graph_layer_type_input;
    pearl_graph_node_input *data = malloc(sizeof(pearl_graph_node_input));
    data->activation_function = pearl_activation_function_type_linear;
    data->num_neurons = num_neurons;
    layer->node_data = data;
    return layer;
}

pearl_graph_layer *pearl_graph_layer_create_output(unsigned int num_neurons)
{
    pearl_graph_layer *layer = pearl_graph_layer_create();
    pearl_graph_node_output *data = malloc(sizeof(pearl_graph_node_output));
    data->activation_function = pearl_activation_function_type_linear;
    data->num_neurons = num_neurons;
    layer->node_type = pearl_graph_layer_type_output;
    layer->node_data = data;
    return layer;
}

pearl_graph_layer *pearl_graph_layer_create_fully_connected(unsigned int num_neurons, unsigned int num_neurons_prev_layer)
{
    pearl_graph_layer *layer = pearl_graph_layer_create();
    layer->node_type = pearl_graph_layer_type_fully_connected;
    pearl_graph_node_fully_connected *data = malloc(sizeof(pearl_graph_node_fully_connected));
    data->activation_function = pearl_activation_function_type_relu;
    data->num_neurons = num_neurons;
    data->biases = pearl_tensor_create(1, data->num_neurons);
    data->weights = pearl_tensor_create(2, data->num_neurons, num_neurons_prev_layer);
    double var = sqrt(2.0 / (data->num_neurons + num_neurons_prev_layer));
    for (unsigned int i = 0; i < data->weights->size[0] * data->weights->size[1]; i++) {
        data->weights->data[i] = pearl_util_rand_norm(0.0, var);
    }
    layer->node_data = data;
    return layer;
}

pearl_graph_layer *pearl_graph_layer_create_dropout(unsigned int num_neurons)
{
    pearl_graph_layer *layer = pearl_graph_layer_create();
    layer->node_type = pearl_graph_layer_type_dropout;
    pearl_graph_node_dropout *data = malloc(sizeof(pearl_graph_node_dropout));
    data->num_neurons = num_neurons;
    if (data->weights == NULL) {
        data->weights = pearl_tensor_create(1, data->num_neurons);
    }
    layer->node_data = data;
    return layer;
}
