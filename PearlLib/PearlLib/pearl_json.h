#ifndef PEARL_JSON_H
#define PEARL_JSON_H

#include <parson.h>
#include <pearl_layer.h>
#include <pearl_network.h>
#include <pearl_tensor.h>
#include <pearl_version.h>

void pearl_json_layer_serialise(pearl_layer *layer, JSON_Object **parent);
void pearl_json_layer_fully_connected_serialise(pearl_layer_data_fully_connected *data, JSON_Object **parent);
void pearl_json_layer_dropout_serialise(pearl_layer_data_dropout *data, JSON_Object **parent);
pearl_layer *pearl_json_layer_deserialise(JSON_Value *json);
PEARL_API void pearl_json_network_serialise(const char *filename, const pearl_network *network);
PEARL_API pearl_network *pearl_json_network_deserialise(const char *filename);
JSON_Value *pearl_json_tensor_serialise(pearl_tensor *tensor);
pearl_tensor *pearl_json_tensor_deserialise(JSON_Value *json);
JSON_Value *pearl_json_version_serialise(pearl_version version);
pearl_version pearl_json_version_deserialise(JSON_Value *json);

#endif // PEARL_JSON_H
