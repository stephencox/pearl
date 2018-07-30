#include <pearl_version.h>

json_object *pearl_version_to_json(pearl_version version)
{
    json_object *json_arr = json_object_new_object();
#ifdef ENV64BIT
    json_object_object_add(json_arr, "major", json_object_new_int64(version.major));
    json_object_object_add(json_arr, "minor", json_object_new_int64(version.minor));
    json_object_object_add(json_arr, "revision", json_object_new_int64(version.revision));
#else
    json_object_object_add(json_arr, "major", json_object_new_int(version.major));
    json_object_object_add(json_arr, "minor", json_object_new_int(version.minor));
    json_object_object_add(json_arr, "revision", json_object_new_int(version.revision));
#endif

    return json_arr;
}
