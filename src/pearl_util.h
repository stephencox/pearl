#ifndef PEARL_UTIL_H
#define PEARL_UTIL_H

#include <math.h>
#include <stdlib.h>
#include <pearl_tensor.h>

float pearl_util_rand_norm(float mu, float sigma);
float pearl_util_accuracy(const pearl_tensor *output, const pearl_tensor *pred);

#endif // PEARL_UTIL_H
