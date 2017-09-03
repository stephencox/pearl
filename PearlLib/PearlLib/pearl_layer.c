#include <pearl_layer.h>

void pearl_layer_initialise(struct pearl_layer *layer, const struct pearl_layer *prev_layer)
{
    if (layer) {
        if (prev_layer) {
            if (!layer->biases) {
                layer->biases = (double *)calloc(layer->neurons, sizeof(double));
            }
            if (!layer->weights) {
                int num_weights = layer->neurons * prev_layer->neurons;
                layer->weights = (double *)calloc(num_weights, sizeof(double));
                double scale = 1.0;
                //https://arxiv.org/abs/1704.08863
                switch (layer->activation_function) {
                    case pearl_activation_function_type_linear:
                        scale = 1.0 / prev_layer->neurons;
                        break;
                    case pearl_activation_function_type_sigmoid:
                        scale = 3.6 / sqrt(prev_layer->neurons);
                        break;
                    case pearl_activation_function_type_tanh:
                        scale = 1.0 / sqrt(prev_layer->neurons);
                        break;
                }
                for (int i = 0; i < num_weights; i++) {
                    layer->weights[i] = rand() / RAND_MAX * scale;
                }
            }
        }
    }
}

void pearl_layer_destroy(struct pearl_layer *layer)
{
    if (layer) {
        if (layer) {
            free(layer->biases);
        }
        if (layer->weights) {
            free(layer->weights);
        }
    }
}
