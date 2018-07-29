TEMPLATE = lib
CONFIG -= app_bundle
CONFIG -= qt

DEFINES += PEARL_LIBRARY

VERSION = 0.1.0.0

SOURCES += pearl_network.c pearl_layer.c \
    pearl_activation_function.c \
    pearl_tensor.c \
    pearl_version.c \
    pearl_loss.c

HEADERS += pearl_activation_function.h \
    pearl_optimiser.h \
    pearl_network.h \
    pearl_layer.h \
    pearl_global.h \
    pearl_tensor.h \
    pearl_loss.h \
    pearl_version.h

# JSON-C
SOURCES += ../../external/json-c/arraylist.c
    ../../external/json-c/debug.c
    ../../external/json-c/json_c_version.c
    ../../external/json-c/json_object.c
    ../../external/json-c/json_object_iterator.c
    ../../external/json-c/json_pointer.c
    ../../external/json-c/json_tokener.c
    ../../external/json-c/json_util.c
    ../../../external/json-c/json_visit.c
    ../../../external/json-c/linkhash.c
    ../../external/json-c/printbuf.c
    ../../external/json-c/random_seed.c
    ../../external/json-c/strerror_override.c
INCLUDEPATH += ../../external/json-c

linux: {
    QMAKE_CFLAGS_RELEASE += -march=native -O3
}

win32: {
    QMAKE_CFLAGS_RELEASE += -Ox
}
