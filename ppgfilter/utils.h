#ifndef utils_h
#define utils_h
#include <stdint.h>

#define RNG_PARAM_A (1103515245)
#define RNG_PARAM_C (12345)

typedef struct {
    uint32_t state;
} rng_t;

uint32_t rng_get_state(rng_t *r);
void rng_set_state(rng_t *r, uint32_t seed);
uint32_t rng_next(rng_t *r);

int absint(int x);

#endif
