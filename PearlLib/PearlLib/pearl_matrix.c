#include <pearl_matrix.h>

PEARL_API pearl_matrix *pearl_matrix_create(int m, int n)
{
    pearl_matrix *result = malloc(sizeof(pearl_matrix));
    result->m = m;
    result->n = n;
    result->data = calloc(m * n, sizeof(double));
    return result;
}

PEARL_API void pearl_matrix_destroy(pearl_matrix *x)
{
    if (x) {
        if (x->data) {
            free(x->data);
        }
        free(x);
    }
}

PEARL_API void pearl_matrix_print(pearl_matrix *x)
{
    printf("%d x %d matrix\n", x->m, x->n);
    for (int i = 0; i < x->m; i++) {
        for (int j = 0; j < x->n; j++) {
            printf("%0.6f\t", x->data[ARRAY_IDX(i, j, x->n)]);
        }
        printf("\n");
    }
}

PEARL_API pearl_matrix *pearl_matrix_copy(const pearl_matrix *x){
    pearl_matrix *result = pearl_matrix_create(x->m, x->n);
    for (int i = 0; i < x->m; i++) {
        for (int j = 0; j < x->n; j++) {
            result->data[ARRAY_IDX(i, j, x->n)] = x->data[ARRAY_IDX(i, j, x->n)];
        }
    }
    return result;
}
