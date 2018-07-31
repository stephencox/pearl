#ifndef PEARL_JSON_H
#define PEARL_JSON_H

#include <parson.h>
#include <pearl_layer.h>
#include <pearl_network.h>
#include <pearl_tensor.h>
#include <pearl_version.h>

JSON_Value *pearl_layer_to_json(pearl_layer *layer);
pearl_layer *pearl_layer_from_json(JSON_Value *json);
PEARL_API void pearl_network_save(const char *filename, const pearl_network *network);
PEARL_API pearl_network *pearl_network_load(const char *filename);
JSON_Value *pearl_tensor_to_json(pearl_tensor *tensor);
pearl_tensor *pearl_tensor_from_json(JSON_Value *json);
JSON_Value *pearl_version_to_json(pearl_version version);
pearl_version pearl_version_from_json(JSON_Value *json);

#endif // PEARL_JSON_H
