#include <sys/time.h>
#include <math.h>
#include <stdlib.h>

#include <openacc.h>

#include "compute.h"

/* ... */

void do_compute(const struct parameters* p, struct results *r)
{
    /* ... */

    struct timeval before, after;
    gettimeofday(&before, NULL);

    int ndev = acc_get_num_devices(acc_device_nvidia);
    printf("Num NVIDIA (%d): %d\n", acc_device_nvidia, ndev);

    /* ... */

    gettimeofday(&after, NULL);
    r->time = (double)(after.tv_sec - before.tv_sec) + 
        (double)(after.tv_usec - before.tv_usec) / 1e6;

    /* ... */
}
