#include <pearl_layer.h>

void pearl_layer_initialise(struct pearl_layer *layer, const struct pearl_layer *prev_layer)
{
    if (layer) {
        if (prev_layer) {
            if (!layer->biases) {
                layer->biases = calloc(layer->neurons, sizeof(double));
            }
            if (!layer->weights) {
                layer->weights = calloc(layer->neurons * prev_layer->neurons, sizeof(double));
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
