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
            printf("%0.0f\t", x->data[ARRAY_IDX(i, j, x->n)]);
        }
        printf("\n");
    }
}

PEARL_API pearl_matrix *pearl_matrix_muliply_transpose_plain(pearl_matrix *x, pearl_matrix *y)
{
    assert(x->n == y->m);
    pearl_matrix *result = pearl_matrix_create(x->m, y->n);
    for (int i = 0; i < x->m; i++) {
        for (int j = 0; j < y->n; j++) {
            double sum = 0;
            for (int k = 0; k < y->m; k++) {
                sum += x->data[ARRAY_IDX(i, k, x->n)] * y->data[ARRAY_IDX(k, j, y->n)];
            }
            result->data[ARRAY_IDX(i, j, result->n)] = sum;
        }
    }
    return result;
}

PEARL_API pearl_matrix *pearl_matrix_muliply_transpose_cblas(pearl_matrix *x, pearl_matrix *y)
{
    assert(x->n == y->m);
    pearl_matrix *result = pearl_matrix_create(x->m, y->n);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, x->m, y->n, x->n, 1, x->data, x->n, y->data, y->n, 1, result->data, result->n);
    return result;
}
