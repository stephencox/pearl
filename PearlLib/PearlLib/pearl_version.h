#ifndef PEARL_VERSION_H
#define PEARL_VERSION_H

#include <pearl_global.h>
#include <parson.h>

//TODO: Handle present version loading

typedef struct {
    unsigned int major;
    unsigned int minor;
    unsigned int revision;
} pearl_version;

JSON_Value *pearl_version_to_json(pearl_version version);
pearl_version pearl_version_from_json(JSON_Value *json);

#endif // PEARL_VERSION_H
