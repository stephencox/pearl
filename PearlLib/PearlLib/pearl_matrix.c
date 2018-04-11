#include <pearl_matrix.h>

PEARL_API struct pearl_matrix *pearl_matrix_create(int m, int n)
{
    struct pearl_matrix *result = malloc(sizeof(struct pearl_matrix));
    result->m = m;
    result->n = n;
    result->data = calloc(m * n, sizeof(double));
    return result;
}

PEARL_API void pearl_matrix_destroy(struct pearl_matrix *x)
{
    if (x) {
        if (x->data) {
            free(x->data);
        }
        free(x);
    }
}

PEARL_API void pearl_matrix_print(struct pearl_matrix *x)
{
    printf("%d x %d matrix\n", x->m, x->n);
    for (int i = 0; i < x->m; i++) {
        for (int j = 0; j < x->n; j++) {
            printf("%0.0f\t", x->data[ARRAY_IDX(i, j, x->n)]);
        }
        printf("\n");
    }
}
