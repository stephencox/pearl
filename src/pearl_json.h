#pragma once

//TODO: Replace with https://github.com/json-c/
#include <parson.h>
#include <pearl_layer.h>
#include <pearl_network.h>
#include <pearl_tensor.h>
#include <pearl_version.h>

void pearl_json_layer_serialise(const pearl_layer *layer, JSON_Object **parent);
void pearl_json_layer_fully_connected_serialise(const pearl_layer_data_fully_connected *data, JSON_Object **parent);
void pearl_json_layer_dropout_serialise(const pearl_layer_data_dropout *data, JSON_Object **parent);
void pearl_json_layer_deserialise(const JSON_Value *, pearl_layer **);
void pearl_json_layer_fully_connected_deserialise(const JSON_Object *, pearl_layer_data_fully_connected **);
void pearl_json_layer_dropout_deserialise(const JSON_Object *, pearl_layer_data_dropout **);
PEARL_API void pearl_json_network_serialise(const char *filename, const pearl_network *network);
PEARL_API pearl_network *pearl_json_network_deserialise(const char *filename);
JSON_Value *pearl_json_tensor_serialise(const pearl_tensor *tensor);
pearl_tensor *pearl_json_tensor_deserialise(const JSON_Value *json);
JSON_Value *pearl_json_version_serialise(const pearl_version version);
pearl_version pearl_json_version_deserialise(const JSON_Value *json);
