#include <pearl_util.h>

float pearl_util_rand_norm(float mu, float sigma)
{
    float U1;
    float U2;
    float W;
    float mult;
    static float X1;
    static float X2;
    static int call = 0;

    if (call == 1) {
        call = !call;
        return (mu + sigma * X2);
    }

    do {
        U1 = -1.0f + ((float) rand() / RAND_MAX) * 2.0f;
        U2 = -1.0f + ((float) rand() / RAND_MAX) * 2.0f;
        W = powf(U1, 2.0f) + powf(U2, 2.0f);
    }
    while (W >= 1 || W == 0);

    mult = sqrtf((-2.0f * logf(W)) / W);
    X1 = U1 * mult;
    X2 = U2 * mult;

    call = !call;

    return (mu + sigma * X1);
}

float pearl_util_accuracy(const pearl_tensor *output, const pearl_tensor *pred)
{
    assert(output->dimension == pred->dimension);
    assert(output->dimension == 1);
    float correct = 0;
    for (unsigned int i = 0; i < output->size[0]; i++) {
        if (output->data[i] < 0.5f && pred->data[i] < 0.5f) {
            correct++;
        }
        if (output->data[i] >= 0.5f && pred->data[i] >= 0.5f) {
            correct++;
        }
    }
    return correct / (float)output->size[0];
}
