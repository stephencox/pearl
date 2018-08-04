#ifndef PEARL_PRINT_H
#define PEARL_PRINT_H

#include <pearl_layer.h>
#include <stdio.h>

void pearl_layer_print(const pearl_layer *layer);
PEARL_API void pearl_tensor_print(const pearl_tensor *x);

#endif // PEARL_PRINT_H
