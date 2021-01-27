#include <time.h>
#include <math.h>
#include <stdlib.h>

#include <pthread.h>

#include "compute.h"

/* ... */

void do_compute(const struct parameters* p, struct results *r)
{
    /* ... */

    struct timespec before, after;
    clock_gettime(CLOCK_MONOTONIC, &before);

    /* ... */

    clock_gettime(CLOCK_MONOTONIC, &after);
    r->time = (double)(after.tv_sec - before.tv_sec) +
              (double)(after.tv_nsec - before.tv_nsec) / 1e9;

    /* ... */
}
