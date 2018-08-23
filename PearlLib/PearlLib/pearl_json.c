#include <pearl_json.h>

void pearl_json_layer_serialise(pearl_layer *layer, JSON_Object **parent)
{
    json_object_set_number((*parent), "type", (double)layer->type);
    json_object_set_number((*parent), "activation", (double)layer->activation.type);
    json_object_set_number((*parent), "num_neurons", (double)layer->num_neurons);
    if (layer->layer_data != NULL) {
        switch (layer->type) {
            case pearl_layer_type_input:
                break;
            case pearl_layer_type_fully_connected:
                pearl_json_layer_fully_connected_serialise((pearl_layer_data_fully_connected *)layer->layer_data, parent);
                break;
            case pearl_layer_type_dropout:
                pearl_json_layer_dropout_serialise((pearl_layer_data_dropout *)layer->layer_data, parent);
                break;
        }
    }
    json_object_set_number((*parent), "num_child_layers", (double)layer->num_child_layers);
    JSON_Value *child_layers = json_value_init_array();
    JSON_Array *child_layers_array = json_value_get_array(child_layers);

    for (unsigned int i = 0; i < layer->num_child_layers; i++) {
        JSON_Value *layer_value = json_value_init_object();
        JSON_Object *layer_object = json_value_get_object(layer_value);
        pearl_json_layer_serialise(layer->child_layers[i], &layer_object);
        json_array_append_value(child_layers_array, layer_value);
    }
    json_object_set_value((*parent), "child_layers", child_layers);
}

void pearl_json_layer_fully_connected_serialise(pearl_layer_data_fully_connected *data, JSON_Object **parent)
{
    json_object_set_value((*parent), "biases", pearl_json_tensor_serialise(data->biases));
    json_object_set_value((*parent), "weights", pearl_json_tensor_serialise(data->weights));
}

void pearl_json_layer_dropout_serialise(pearl_layer_data_dropout *data, JSON_Object **parent)
{
    json_object_set_number((*parent), "rate", data->rate);
}

void pearl_json_layer_deserialise(JSON_Value *json, pearl_layer **parent)
{
    JSON_Object *obj = json_value_get_object(json);
    pearl_layer_type layer_type = (pearl_layer_type)json_object_get_number(obj, "type");
    pearl_activation_type layer_activation_type = (pearl_activation_type)json_object_get_number(obj, "activation");
    unsigned int layer_num_neurons = (unsigned int)json_object_get_number(obj, "num_neurons");
    unsigned int layer_num_child_layers = (unsigned int)json_object_get_number(obj, "num_child_layers");
    JSON_Value *layer_childs = json_object_get_value(obj, "child_layers");
    JSON_Array *layer_childs_array = json_value_get_array(layer_childs);
    switch (layer_type) {
        case pearl_layer_type_input:
            (*parent) = pearl_layer_create_input(layer_num_neurons);
            break;
        case pearl_layer_type_fully_connected:
            (*parent) = pearl_layer_create_fully_connected_blank(layer_num_neurons);
            pearl_json_layer_fully_connected_deserialise(obj, (pearl_layer_data_fully_connected **) & (*parent)->layer_data);
            break;
        case pearl_layer_type_dropout:
            (*parent) = pearl_layer_create_dropout(layer_num_neurons);
            pearl_json_layer_dropout_deserialise(obj, (pearl_layer_data_dropout **) & (*parent)->layer_data);
            break;
    }
    (*parent)->activation = pearl_activation_create(layer_activation_type);
    for (unsigned int i = 0; i < layer_num_child_layers; i++) {
        pearl_layer *child;
        pearl_json_layer_deserialise(json_array_get_value(layer_childs_array, i), &child);
        pearl_layer_add_child(parent, &child);
    }
}

void pearl_json_layer_fully_connected_deserialise(JSON_Object *obj, pearl_layer_data_fully_connected **data)
{
    JSON_Value *biases = json_object_get_value(obj, "biases");
    if (biases != NULL) {
        (*data)->biases = pearl_json_tensor_deserialise(biases);
    }
    JSON_Value *weights = json_object_get_value(obj, "weights");
    if (weights != NULL) {
        (*data)->weights = pearl_json_tensor_deserialise(weights);
    }
}

void pearl_json_layer_dropout_deserialise(JSON_Object *obj, pearl_layer_data_dropout **data)
{
    (*data)->rate = json_object_get_number(obj, "rate");
}

PEARL_API void pearl_json_network_serialise(const char *filename, const pearl_network *network)
{
    JSON_Value *root_value = json_value_init_object();
    JSON_Object *root_object = json_value_get_object(root_value);
    json_object_set_value(root_object, "version", pearl_json_version_serialise(network->version));
    json_object_set_number(root_object, "loss_type", (double)network->loss.type);
    json_object_set_number(root_object, "optimiser", (double)network->optimiser);
    json_object_set_number(root_object, "learning_rate", network->learning_rate);
    JSON_Value *layers_value = json_value_init_object();
    JSON_Object *layers_object = json_value_get_object(layers_value);
    pearl_json_layer_serialise(network->input_layer, &layers_object);
    json_object_set_value(root_object, "input_layer", layers_value);
    json_serialize_to_file(root_value, filename);
    json_value_free(root_value);
}

//TODO: Handle errors in loading
PEARL_API pearl_network *pearl_json_network_deserialise(const char *filename)
{
    pearl_network *network;
    JSON_Value *value = json_parse_file(filename);
    JSON_Object *obj = json_value_get_object(value);
    JSON_Value *json_network_version = json_object_get_value(obj, "version");
    if (json_network_version == NULL) {
        return NULL;
    }
    pearl_version version = pearl_json_version_deserialise(json_network_version);
    pearl_loss_type loss_type = (pearl_loss_type)json_object_get_number(obj, "loss_type");
    pearl_optimiser optimiser = (pearl_optimiser)json_object_get_number(obj, "optimiser");
    double learning_rate = json_object_get_number(obj, "learning_rate");

    JSON_Value *json_network_input_layer = json_object_get_value(obj, "input_layer");
    if (json_network_input_layer == NULL) {
        return NULL;
    }
    pearl_layer *input;
    pearl_json_layer_deserialise(json_network_input_layer, &input);

    network = pearl_network_create();
    network->version = version;
    network->learning_rate = learning_rate;
    network->loss = pearl_loss_create(loss_type);
    network->optimiser = optimiser;
    network->input_layer = input;
    //TODO: find output

    json_value_free(value);

    return network;
}

JSON_Value *pearl_json_tensor_serialise(pearl_tensor *tensor)
{
    JSON_Value *value = json_value_init_object();
    JSON_Object *obj = json_value_get_object(value);
    json_object_set_number(obj, "dimension", tensor->dimension);
    JSON_Value *size = json_value_init_array();
    JSON_Array *size_array = json_value_get_array(size);
    unsigned int num_data = 1;
    for (unsigned int i = 0 ; i < tensor->dimension; i++) {
        json_array_append_number(size_array, tensor->size[i]);
        num_data *= tensor->size[i];
    }
    json_object_set_value(obj, "size", size);
    JSON_Value *data = json_value_init_array();
    JSON_Array *data_array = json_value_get_array(data);
    for (unsigned int i = 0; i < num_data; i++) {
        json_array_append_number(data_array, tensor->data[i]);
    }
    json_object_set_value(obj, "data", data);
    return value;
}

pearl_tensor *pearl_json_tensor_deserialise(JSON_Value *json)
{
    JSON_Object *obj = json_value_get_object(json);
    pearl_tensor *tensor = malloc(sizeof(pearl_tensor));
    tensor->dimension = (unsigned int)json_object_get_number(obj, "dimension");
    JSON_Value *tensor_size_array = json_object_get_value(obj, "size");
    if (tensor_size_array == NULL) {
        pearl_tensor_destroy(&tensor);
        return NULL;
    }
    tensor->size = calloc(tensor->dimension, sizeof(unsigned int));
    unsigned int num_data = 1;
    JSON_Array *size_array = json_value_get_array(tensor_size_array);
    if (size_array == NULL) {
        pearl_tensor_destroy(&tensor);
        return NULL;
    }
    for (unsigned int i = 0; i < tensor->dimension; i++) {
        tensor->size[i] = (unsigned int)json_array_get_number(size_array, i);
        num_data *= tensor->size[i];
    }
    JSON_Value *tensor_data_array = json_object_get_value(obj, "data");
    if (tensor_data_array == NULL) {
        pearl_tensor_destroy(&tensor);
        return NULL;
    }
    tensor->data = calloc(num_data, sizeof(double));
    JSON_Array *data_array = json_value_get_array(tensor_data_array);
    if (data_array == NULL) {
        pearl_tensor_destroy(&tensor);
        return NULL;
    }
    for (unsigned int i = 0; i < num_data; i++) {
        tensor->data[i] = json_array_get_number(data_array, i);
    }

    return tensor;
}

JSON_Value *pearl_json_version_serialise(pearl_version version)
{
    JSON_Value *root_value = json_value_init_object();
    JSON_Object *root_object = json_value_get_object(root_value);
    json_object_set_number(root_object, "major", version.major);
    json_object_set_number(root_object, "minor", version.minor);
    json_object_set_number(root_object, "revision", version.revision);
    return root_value;
}

pearl_version pearl_json_version_deserialise(JSON_Value *json)
{
    JSON_Object *obj = json_value_get_object(json);
    pearl_version version;
    version.major = (unsigned int)json_object_get_number(obj, "major");
    version.minor = (unsigned int)json_object_get_number(obj, "minor");
    version.revision = (unsigned int) json_object_get_number(obj, "revision");
    return version;
}
