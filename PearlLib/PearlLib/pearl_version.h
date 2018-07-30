#ifndef PEARL_VERSION_H
#define PEARL_VERSION_H

#include <pearl_global.h>
#include <json.h>

typedef struct {
    unsigned int major;
    unsigned int minor;
    unsigned int revision;
} pearl_version;

json_object *pearl_version_to_json(pearl_version version);

#endif // PEARL_VERSION_H
