#ifndef PEARL_PRINT_H
#define PEARL_PRINT_H

#include <pearl_layer.h>
#include <pearl_tensor.h>
#include <stdio.h>

void pearl_print_layer(const pearl_layer *layer);
PEARL_API void pearl_print_tensor(const pearl_tensor *x);

#endif // PEARL_PRINT_H
