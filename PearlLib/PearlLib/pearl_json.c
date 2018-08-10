#include <pearl_json.h>

JSON_Value *pearl_json_layer_serialise(pearl_layer *layer)
{
    JSON_Value *value = json_value_init_object();
    JSON_Object *obj = json_value_get_object(value);
    json_object_set_value(obj, "version", pearl_json_version_serialise(layer->version));
    json_object_set_number(obj, "activation", (double)layer->activation_function);
    json_object_set_number(obj, "neurons", layer->num_neurons);
    json_object_set_number(obj, "type", (double)layer->type);
    json_object_set_value(obj, "biases", pearl_json_tensor_serialise(layer->biases));
    json_object_set_value(obj, "weights", pearl_json_tensor_serialise(layer->weights));
    return value;
}

pearl_layer *pearl_json_layer_deserialise(JSON_Value *json)
{
    JSON_Object *obj = json_value_get_object(json);
    JSON_Value *layer_version = json_object_get_value(obj, "version");
    if (layer_version == NULL) {
        return NULL;
    }
    pearl_layer *layer = malloc(sizeof(pearl_layer));
    layer->version = pearl_json_version_deserialise(layer_version);
    layer->activation_function = (pearl_activation_function_type)json_object_get_number(obj, "activation");
    layer->num_neurons = (unsigned int)json_object_get_number(obj, "neurons");
    layer->type = (pearl_layer_type)json_object_get_number(obj, "type");
    layer->biases = pearl_json_tensor_deserialise(json_object_get_value(obj, "biases"));
    layer->weights = pearl_json_tensor_deserialise(json_object_get_value(obj, "weights"));
    return layer;
}

PEARL_API void pearl_json_network_serialise(const char *filename, const pearl_network *network)
{
    JSON_Value *root_value = json_value_init_object();
    JSON_Object *root_object = json_value_get_object(root_value);
    json_object_set_value(root_object, "version", pearl_json_version_serialise(network->version));
    json_object_set_number(root_object, "num_input", network->num_input);
    json_object_set_number(root_object, "num_output", network->num_output);
    json_object_set_number(root_object, "num_layers", network->num_layers);
    json_object_set_number(root_object, "loss_type", (double)network->loss.type);
    json_object_set_number(root_object, "optimiser", (double)network->optimiser);
    json_object_set_number(root_object, "learning_rate", network->learning_rate);
    JSON_Value *layers = json_value_init_array();
    JSON_Array *layers_array = json_value_get_array(layers);
    for (unsigned int i = 0; i < network->num_layers; i++) {
        json_array_append_value(layers_array, pearl_json_layer_serialise(network->layers[i]));
    }
    json_object_set_value(root_object, "layers", layers);
    json_serialize_to_file(root_value, filename);
    json_value_free(root_value);
}

//TODO: Handle errors in loading
PEARL_API pearl_network *pearl_json_network_deserialise(const char *filename)
{
    pearl_network *network;
    JSON_Value *value = json_parse_file(filename);
    JSON_Object *obj = json_value_get_object(value);

    // VERSION
    JSON_Value *json_network_version = json_object_get_value(obj, "version");
    if (json_network_version == NULL) {
        return NULL;
    }
    pearl_version version = pearl_json_version_deserialise(json_network_version);
    unsigned int num_input = (unsigned int)json_object_get_number(obj, "num_input");
    unsigned int num_output = (unsigned int)json_object_get_number(obj, "num_output");
    unsigned int num_layers = (unsigned int)json_object_get_number(obj, "num_layers");
    pearl_loss_type loss_type = (pearl_loss_type)json_object_get_number(obj, "loss_type");
    pearl_optimiser optimiser = (pearl_optimiser)json_object_get_number(obj, "optimiser");
    double learning_rate = json_object_get_number(obj, "learning_rate");

    JSON_Array *layer_array = json_object_get_array(obj, "layers");
    if (layer_array == NULL) {
        return NULL;
    }

    network = pearl_network_create(num_input, num_output);
    network->version = version;
    network->learning_rate = learning_rate;
    network->loss = pearl_loss_create(loss_type);
    network->num_layers = num_layers;
    network->optimiser = optimiser;
    network->layers = calloc(network->num_layers, sizeof(pearl_layer *));
    for (unsigned int i = 0; i < network->num_layers; i++) {
        JSON_Value *item = json_array_get_value(layer_array, i);
        if (item != NULL) {
            network->layers[i] = pearl_json_layer_deserialise(item);
        }
        else {
            pearl_network_destroy(&network);
            return NULL;
        }
    }
    json_value_free(value);

    return network;
}

JSON_Value *pearl_json_tensor_serialise(pearl_tensor *tensor)
{
    JSON_Value *value = json_value_init_object();
    JSON_Object *obj = json_value_get_object(value);
    json_object_set_value(obj, "version", pearl_json_version_serialise(tensor->version));
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
    JSON_Value *tensor_version = json_object_get_value(obj, "version");
    if (tensor_version == NULL) {
        return NULL;
    }
    pearl_tensor *tensor = malloc(sizeof(pearl_tensor));
    tensor->version = pearl_json_version_deserialise(tensor_version);
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
