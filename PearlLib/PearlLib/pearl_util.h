#ifndef PEARL_UTIL_H
#define PEARL_UTIL_H

#include <math.h>
#include <stdlib.h>
#include <pearl_tensor.h>

double pearl_util_rand_norm(double mu, double sigma);
double pearl_util_accuracy(const pearl_tensor *output, pearl_tensor *pred);

#endif // PEARL_UTIL_H
