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

pearl_version pearl_version_from_json(json_object *json)
{
    pearl_version version;
    json_object *major = json_object_object_get(json, "major");
    if (major != NULL) {
#ifdef ENV64BIT
        version.major = json_object_get_int64(major);
#else
        version.major = json_object_get_int(major);
#endif
    }
    else {
        version.major = 0;
    }

    json_object *minor = json_object_object_get(json, "minor");
    if (minor != NULL) {
#ifdef ENV64BIT
        version.minor = json_object_get_int64(minor);
#else
        version.minor = json_object_get_int(minor);
#endif
    }
    else {
        version.minor = 0;
    }

    json_object *revision = json_object_object_get(json, "revision");
    if (revision != NULL) {
#ifdef ENV64BIT
        version.revision = json_object_get_int64(revision);
#else
        version.revision = json_object_get_int(revision);
#endif
    }
    else {
        version.revision = 0;
    }

    return version;
}
