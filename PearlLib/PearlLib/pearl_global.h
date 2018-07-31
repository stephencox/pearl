#ifndef PEARL_GLOBAL_H
#define PEARL_GLOBAL_H

#if _WIN32 || _WIN64
    #if defined(PEARL_LIBRARY)
        #define PEARL_API __declspec(dllexport)
    #else
        #define PEARL_API __declspec(dllimport)
    #endif
#else
    #define PEARL_API
#endif

#endif // PEARL_GLOBAL_H
