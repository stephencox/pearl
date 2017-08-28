TEMPLATE = lib
CONFIG -= app_bundle
CONFIG -= qt

DEFINES += PEARL_LIBRARY

SOURCES += \
    pearl_network.c \
    pearl_layer.c

HEADERS += \
    pearl_activation_function.h \
    pearl_optimiser.h \
    pearl_network.h \
    pearl_layer.h \
    pearl_global.h
