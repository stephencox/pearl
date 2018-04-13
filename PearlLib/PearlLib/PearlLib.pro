TEMPLATE = lib
CONFIG -= app_bundle
CONFIG -= qt

DEFINES += PEARL_LIBRARY

VERSION = 0.1.0.0

SOURCES += pearl_network.c pearl_layer.c pearl_matrix.c \
    pearl_vector.c \
    pearl_activation_function.c

HEADERS += pearl_activation_function.h pearl_optimiser.h pearl_network.h pearl_layer.h pearl_global.h pearl_matrix.h \
    pearl_vector.h

linux: {
    QMAKE_CFLAGS_RELEASE += -march=native -O3
}

win32: {
    QMAKE_CFLAGS_RELEASE += -Ox
}
