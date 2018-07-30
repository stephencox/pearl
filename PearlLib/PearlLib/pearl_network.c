#include <pearl_network.h>

PEARL_API pearl_network *pearl_network_create(const unsigned int num_input, const unsigned int num_output)
{
    srand((unsigned int)time(NULL));
    pearl_network *network = malloc(sizeof(pearl_network));
    network->num_layers = 0;
    network->optimiser = pearl_optimiser_sgd;
    network->loss_type = pearl_loss_binary_cross_entropy;
    network->learning_rate = 1e-3;
    network->num_input = num_input;
    network->num_output = num_output;
    network->layers = NULL;
    network->version.major = PEARL_NETWORK_VERSION_MAJOR;
    network->version.minor = PEARL_NETWORK_VERSION_MINOR;
    network->version.revision = PEARL_NETWORK_VERSION_REVISION;
    return network;
}

PEARL_API void pearl_network_destroy(pearl_network **network)
{
    if (*network != NULL) {
        if ((*network)->layers != NULL) {
            for (int i = 0; i < (*network)->num_layers; i++) {
                pearl_layer_destroy(&(*network)->layers[i]);
            }
            free((*network)->layers);
            (*network)->layers = NULL;
        }
        free(*network);
        *network = NULL;
    }
}

PEARL_API void pearl_network_save(const char *filename, const pearl_network *network)
{
    json_object *json_obj = json_object_new_object();

    json_object_object_add(json_obj, "version", pearl_version_to_json(network->version));
#ifdef ENV64BIT
    json_object_object_add(json_obj, "num_input", json_object_new_int64(network->num_input));
    json_object_object_add(json_obj, "num_output", json_object_new_int64(network->num_output));
    json_object_object_add(json_obj, "num_layers", json_object_new_int64(network->num_layers));
    json_object_object_add(json_obj, "loss_type", json_object_new_int64((int)network->loss_type));
    json_object_object_add(json_obj, "pearl_optimiser", json_object_new_int64((int)network->optimiser));
#else
    json_object_object_add(json_obj, "num_input", json_object_new_int(network->num_input));
    json_object_object_add(json_obj, "num_output", json_object_new_int(network->num_output));
    json_object_object_add(json_obj, "num_layers", json_object_new_int(network->num_layers));
    json_object_object_add(json_obj, "loss_type", json_object_new_int((int)network->loss_type));
    json_object_object_add(json_obj, "pearl_optimiser", json_object_new_int((int)network->optimiser));
#endif
    json_object_object_add(json_obj, "learning_rate", json_object_new_double(network->learning_rate));
    json_object *json_network_layers = json_object_new_array();
    for (int i = 0; i < network->num_layers; i++) {
        json_object_array_add(json_network_layers, pearl_layer_to_json(network->layers[i]));
    }
    json_object_object_add(json_obj, "layers",json_network_layers);
    json_object_to_file(filename, json_obj);
}

PEARL_API pearl_network *pearl_network_load(const char *filename)
{
    FILE *f = fopen(filename, "rb");
    if (f == NULL) {
        fprintf(stderr, "Load failed: Error opening file!\n");
        return NULL;
    }
    pearl_network *network = malloc(sizeof(pearl_network));
    network->layers = NULL;
    size_t count = 0;
    count = fread(&network->version, sizeof(pearl_version), 1, f);
    if(count!=1){
        fprintf(stderr, "Load failed: Error reading network version!\n");
        pearl_network_destroy(&network);
        fclose(f);
        return NULL;
    }

    count = fread(&network->num_input, sizeof(unsigned int), 1, f);
    if(count!=1){
        fprintf(stderr, "Load failed: Error reading num_input!\n");
        pearl_network_destroy(&network);
        fclose(f);
        return NULL;
    }

    count = fread(&network->num_output, sizeof(unsigned int), 1, f);
    if(count!=1){
        fprintf(stderr, "Load failed: Error reading num_output!\n");
        pearl_network_destroy(&network);
        fclose(f);
        return NULL;
    }

    count = fread(&network->num_layers, sizeof(unsigned int), 1, f);
    if(count!=1){
        fprintf(stderr, "Load failed: Error reading num_layers!\n");
        pearl_network_destroy(&network);
        fclose(f);
        return NULL;
    }

    count = fread(&network->learning_rate, sizeof(double), 1, f);
    if(count!=1){
        fprintf(stderr, "Load failed: Error reading learning_rate!\n");
        pearl_network_destroy(&network);
        fclose(f);
        return NULL;
    }

    count = fread(&network->loss_type, sizeof(pearl_loss), 1, f);
    if(count!=1){
        fprintf(stderr, "Load failed: Error reading loss_type!\n");
        pearl_network_destroy(&network);
        fclose(f);
        return NULL;
    }

    count = fread(&network->optimiser, sizeof(pearl_optimiser), 1, f);
    if(count!=1){
        fprintf(stderr, "Load failed: Error reading optimiser!\n");
        pearl_network_destroy(&network);
        fclose(f);
        return NULL;
    }

    network->layers = calloc(network->num_layers, sizeof(pearl_layer*));
    for (int i = 0; i < network->num_layers; i++) {
        network->layers[i] = NULL;
    }

    for (int i = 0; i < network->num_layers; i++) {

        pearl_layer *layer = malloc(sizeof(pearl_layer));
        layer->weights = NULL;
        layer->biases = NULL;
        count = fread(&layer->version, sizeof(pearl_version), 1, f);
        if(count!=1){
            fprintf(stderr, "Load failed: Error reading layer %d version!\n", i+1);
            pearl_network_destroy(&network);
            pearl_layer_destroy(&layer);
            fclose(f);
            return NULL;
        }

        count = fread(&layer->activation_function, sizeof(pearl_activation_function_type), 1, f);
        if(count!=1){
            fprintf(stderr, "Load failed: Error reading layer %d activation_function!\n", i+1);
            pearl_network_destroy(&network);
            pearl_layer_destroy(&layer);
            fclose(f);
            return NULL;
        }

        count = fread(&layer->neurons, sizeof(unsigned int), 1, f);
        if(count!=1){
            fprintf(stderr, "Load failed: Error reading layer %d neurons!\n", i+1);
            pearl_network_destroy(&network);
            pearl_layer_destroy(&layer);
            fclose(f);
            return NULL;
        }

        count = fread(&layer->type, sizeof(pearl_layer_type), 1, f);
        if(count!=1){
            fprintf(stderr, "Load failed: Error reading layer %d type!\n", i+1);
            pearl_network_destroy(&network);
            pearl_layer_destroy(&layer);
            fclose(f);
            return NULL;
        }

        layer->weights = pearl_tensor_load(f);
        if(layer->weights == NULL){
            fprintf(stderr, "Load failed: Error reading weights!\n");
            pearl_network_destroy(&network);
            pearl_layer_destroy(&layer);
            fclose(f);
            return NULL;
        }

        layer->biases = pearl_tensor_load(f);
        if(layer->biases == NULL){
            fprintf(stderr, "Load failed: Error reading biases!\n");
            pearl_network_destroy(&network);
            pearl_layer_destroy(&layer);
            fclose(f);
            return NULL;
        }

        network->layers[i] = layer;
    }

    fclose(f);

    return network;
}

PEARL_API void pearl_network_layer_add(pearl_network **network, const pearl_layer_type type, const int neurons, const pearl_activation_function_type activation_function)
{
    (*network)->num_layers++;
    if ((*network)->num_layers > 1) {
        (*network)->layers = realloc((*network)->layers, (*network)->num_layers * sizeof(pearl_layer*)); //TODO: error checking
    }
    else {
        (*network)->layers = calloc(1, sizeof(pearl_layer*));
    }
    pearl_layer *layer = malloc(sizeof(pearl_layer));
    layer->type = type;
    layer->neurons = neurons;
    layer->activation_function = activation_function;
    layer->weights = NULL;
    layer->biases = NULL;
    //layer->dropout_rate = 0.0;
    layer->version.major = PEARL_LAYER_VERSION_MAJOR;
    layer->version.minor = PEARL_LAYER_VERSION_MINOR;
    layer->version.revision = PEARL_LAYER_VERSION_REVISION;
    (*network)->layers[(*network)->num_layers - 1] = layer;
}

PEARL_API void pearl_network_layer_add_output(pearl_network **network, const pearl_activation_function_type activation_function)
{
    pearl_network_layer_add(network, pearl_layer_type_fully_connect, (*network)->num_output, activation_function);
}

//PEARL_API void pearl_network_layer_add_dropout(pearl_network **network, const int neurons, const pearl_activation_function_type activation_function, const double dropout_rate)
//{
//    pearl_network_layer_add(network, pearl_layer_type_dropout, neurons, activation_function);
//    pearl_network *network_p = (*network);
//    network_p->layers[network_p->num_layers - 1]->dropout_rate = dropout_rate;
//}

PEARL_API void pearl_network_layer_add_fully_connect(pearl_network **network, const int neurons, const pearl_activation_function_type activation_function)
{
    pearl_network_layer_add(network, pearl_layer_type_fully_connect, neurons, activation_function);
}

PEARL_API void pearl_network_layers_initialise(pearl_network **network)
{
    if ((*network)->layers != NULL) {
        for (int i = 0; i < (*network)->num_layers; i++) {
            int num_neurons_next_layer = (i < (*network)->num_layers-1 ? (*network)->layers[i + 1]->neurons : (*network)->num_output);
            pearl_layer_initialise(&(*network)->layers[i], num_neurons_next_layer);
        }
    }
}

PEARL_API void pearl_network_train_epoch(pearl_network **network, const pearl_tensor *input, const pearl_tensor *output)
{
    pearl_tensor **z = calloc((*network)->num_layers, sizeof(pearl_tensor *));
    pearl_tensor **a = calloc((*network)->num_layers + 1, sizeof(pearl_tensor *));
    // Forward
    a[0] = pearl_tensor_create(2, input->size[1], input->size[0]);
    for (unsigned int i = 0; i < input->size[0]; i++) {
        for (unsigned int j = 0; j < input->size[1]; j++) {
            a[0]->data[ARRAY_IDX_2D(j, i, a[0]->size[1])] = input->data[ARRAY_IDX_2D(i, j, input->size[1])];
        }
    }

    for (int i = 0; i < (*network)->num_layers; i++) {
        printf("Layer %d\n", i);
        assert(z[i] == NULL);
        z[i] = pearl_tensor_create(2, (*network)->layers[i]->weights->size[0], a[i]->size[1]);
        assert(a[i + 1] == NULL);
        a[i + 1] = pearl_tensor_create(2, (*network)->layers[i]->weights->size[0], a[i]->size[1]);
        pearl_layer_forward(&(*network)->layers[i], a[i], &z[i], &a[i + 1]);
    }
    // Cost
    pearl_tensor *al = a[(*network)->num_layers];
    double cost = 0.0;
    switch ((*network)->loss_type) {
    case pearl_loss_binary_cross_entropy:
        cost = pearl_loss_binary_cross_entropy_cost(output, al);
        break;
    default:
        printf("Invalid loss function\n");
        break;
    }
    printf("Loss: %f\n", cost);

    //Backward
    pearl_tensor **dw = calloc((*network)->num_layers, sizeof(pearl_tensor *));
    pearl_tensor **db = calloc((*network)->num_layers, sizeof(pearl_tensor *));
    pearl_tensor **dz = calloc((*network)->num_layers, sizeof(pearl_tensor *));
    for (int i = (*network)->num_layers - 1; i >= 0; i--) {
        if (i == (*network)->num_layers - 1) {
            assert(dz[i] == NULL);
            dz[i] = pearl_tensor_create(2, output->size[1], output->size[0]);
            for (unsigned int j = 0; j < output->size[1]; j++) {
                for (unsigned int x = 0; x < output->size[0]; x++) {
                    assert(ARRAY_IDX_2D(j, x, output->size[0]) < output->size[0]*output->size[1]);
                    assert(ARRAY_IDX_2D(j, x, dz[i]->size[1]) < dz[i]->size[0]*dz[i]->size[1]);
                    assert(ARRAY_IDX_2D(j, x, al->size[1]) < al->size[0]*al->size[1]);
                    double temp = output->data[ARRAY_IDX_2D(j, x, output->size[0])];
                    if (temp > 0.0) {
                        dz[i]->data[ARRAY_IDX_2D(j, x, dz[i]->size[1])] = - (output->data[ARRAY_IDX_2D(j, x, output->size[0])] / al->data[ARRAY_IDX_2D(j, x, al->size[1])]);
                    }
                    else {
                        dz[i]->data[ARRAY_IDX_2D(j, x, dz[i]->size[1])] = ((1.0 - output->data[ARRAY_IDX_2D(j, x, output->size[0])]) / (1.0 - al->data[ARRAY_IDX_2D(j, x, al->size[1])]));
                    }
                }
            }
            printf("dZ at output\n");
            pearl_tensor_print(dz[i]);
        }
        if (i > 0) {
            assert(dw[i] == NULL);
            dw[i] = pearl_tensor_create(2, dz[i]->size[0], a[i]->size[0]);
            assert(db[i] == NULL);
            db[i] = pearl_tensor_create(1, dz[i]->size[0]);
            assert(dz[i - 1] == NULL);
            dz[i - 1] = pearl_tensor_create(2, (*network)->layers[i]->weights->size[1], dz[i]->size[1]);
            pearl_layer_backward((*network)->layers[i], (*network)->layers[i - 1]->activation_function, dz[i], a[i], z[i], &dw[i], &db[i], &dz[i - 1]);
        }
        else {
            assert(dw[i] == NULL);
            dw[i] = pearl_tensor_create(2, dz[i]->size[0], a[i]->size[0]);
            assert(db[i] == NULL);
            db[i] = pearl_tensor_create(1, dz[i]->size[0]);
            pearl_layer_backward_weights_biases(dz[i], a[i], &dw[i], &db[i]);
        }
    }
    //Update
    for (int i = 0; i < (*network)->num_layers; i++) {
        pearl_layer_update((*network)->layers[i], dw[i], db[i], (*network)->learning_rate);
    }
    // Clean
    for (int i = 0; i < (*network)->num_layers - 1; i++) {
        pearl_tensor_destroy(&a[i]);
        pearl_tensor_destroy(&z[i]);
        pearl_tensor_destroy(&dw[i]);
        pearl_tensor_destroy(&db[i]);
        pearl_tensor_destroy(&dz[i]);
    }
    free(a);
    free(z);
    free(dw);
    free(db);
    free(dz);
}
