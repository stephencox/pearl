#ifndef PEARL_VERSION_H
#define PEARL_VERSION_H

#include <pearl_global.h>
#include <json.h>

//TODO: Handle present version loading

typedef struct {
    unsigned int major;
    unsigned int minor;
    unsigned int revision;
} pearl_version;

json_object *pearl_version_to_json(pearl_version version);
pearl_version pearl_version_from_json(json_object *json);

#endif // PEARL_VERSION_H
