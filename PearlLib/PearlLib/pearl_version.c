#include <pearl_version.h>

JSON_Value *pearl_version_to_json(pearl_version version)
{
    JSON_Value *root_value = json_value_init_object();
    JSON_Object *root_object = json_value_get_object(root_value);
    json_object_set_number(root_object, "major", version.major);
    json_object_set_number(root_object, "minor", version.minor);
    json_object_set_number(root_object, "revision", version.revision);
    return root_value;
}

pearl_version pearl_version_from_json(JSON_Value *json)
{
    JSON_Object *obj = json_value_get_object(json);
    pearl_version version;
    version.major = json_object_get_number(obj, "major");
    version.minor = json_object_get_number(obj, "minor");
    version.revision = json_object_get_number(obj, "revision");
    return version;
}
