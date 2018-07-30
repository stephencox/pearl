#ifndef PEARL_GLOBAL_H
#define PEARL_GLOBAL_H

#if _WIN32 || _WIN64
    #if defined(PEARL_LIBRARY)
        #define PEARL_API __declspec(dllexport)
    #else
        #define PEARL_API __declspec(dllimport)
    #endif

    #if _WIN64
        #define ENV64BIT
    #else
        #define ENV32BIT
    #endif
#else
    #define PEARL_API
#endif

#if __GNUC__
    #if __x86_64__ || __ppc64__
        #define ENV64BIT
    #else
        #define ENV32BIT
    #endif
#endif

#endif // PEARL_GLOBAL_H
