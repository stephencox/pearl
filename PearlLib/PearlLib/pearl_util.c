#include <pearl_util.h>

double pearl_util_rand_norm(double mu, double sigma)
{
    double U1;
    double U2;
    double W;
    double mult;
    static double X1;
    static double X2;
    static int call = 0;

    if (call == 1) {
        call = !call;
        return (mu + sigma * X2);
    }

    do {
        U1 = -1 + ((double) rand() / RAND_MAX) * 2;
        U2 = -1 + ((double) rand() / RAND_MAX) * 2;
        W = pow(U1, 2) + pow(U2, 2);
    }
    while (W >= 1 || W == 0);

    mult = sqrt((-2 * log(W)) / W);
    X1 = U1 * mult;
    X2 = U2 * mult;

    call = !call;

    return (mu + sigma * X1);
}

double pearl_util_accuracy(const pearl_tensor *output, pearl_tensor *pred)
{
    assert(output->dimension == pred->dimension);
    assert(output->dimension == 1);
    unsigned int correct = 0;
    for (unsigned int i = 0; i < output->size[0]; i++) {
        if (output->data[i] < 0.5 && pred->data[i] < 0.5) {
            correct++;
        }
        if (output->data[i] >= 0.5 && pred->data[i] >= 0.5) {
            correct++;
        }
    }
    return 1.0 * correct / output->size[0];
}
