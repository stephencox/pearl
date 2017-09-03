#include <stdio.h>
#include <pearl_network.h>

#include <pearl_matrix.h>
int main()
{
    /*struct pearl_network *network = pearl_network_create();
    pearl_network_layer_add_input(network, 3000);
    pearl_network_layer_add_fully_connect(network, 3000, pearl_activation_function_type_tanh);
    pearl_network_layer_add_output(network, 1, pearl_activation_function_type_linear);
    pearl_network_layers_initialise(network);


    pearl_network_destroy(network);*/

#define XM 1000
#define XN 1000
#define YM 1000
#define YN 1000

    printf("Creating data\n");
    pearl_matrix *x = pearl_matrix_create(XM, XN);
    for (int i = 0; i < x->m; i++) {
        for (int j = 0; j < x->n; j++) {
            x->data[ARRAY_IDX(i, j, x->n)] = (i + 1) * 10 + j + 1;
        }
    }

    pearl_matrix *y = pearl_matrix_create(YM, YN);
    for (int i = 0; i < y->m; i++) {
        for (int j = 0; j < y->n; j++) {
            y->data[ARRAY_IDX(i, j, y->n)] = (i + 1) * 10 + j + 1;
        }
    }

    printf("Running plain\n");
    clock_t begin_a = clock();
    pearl_matrix *c = pearl_matrix_muliply_transpose_plain(x, y);
    clock_t end_a = clock();

    printf("Running cblas\n");
    clock_t begin_b = clock();
    pearl_matrix *d = pearl_matrix_muliply_transpose_cblas(x, y);
    clock_t end_b = clock();

    double time_spent_a = (double)(end_a - begin_a) / CLOCKS_PER_SEC;
    double time_spent_b = (double)(end_b - begin_b) / CLOCKS_PER_SEC;

    printf("a=%0.6f\nb=%0.6f\nspeedup_b=%0.2f x\n", time_spent_a * 1000, time_spent_b * 1000, time_spent_a / time_spent_b);

    free(x);
    free(y);
    pearl_matrix_destroy(c);
    pearl_matrix_destroy(d);
    return 0;
}
