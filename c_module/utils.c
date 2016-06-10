#include "utils.h"

void rng_set_state(rng_t *r, uint32_t seed){
    r->state = seed;
}

uint32_t rng_get_state(rng_t *r){
    return r->state;
}

uint32_t rng_next(rng_t *r){
    r->state = (r->state * RNG_PARAM_A + RNG_PARAM_C);
    return r->state;
}
