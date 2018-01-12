#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <string.h>

#if defined(MATMUL_ACC) | defined(MATMUL_ACC_ADV)
#include <openacc.h>
#endif

#ifdef MATMUL_OMP
#include <omp.h>
#endif

int debug = 0;

void matmul(float *a, float *b, float *x, int len, int iters);

#ifdef MATMUL_SEQ
void matmul(float *a, float *b, float *x, int len, int iters)
{
    /* Loop such that it may consume a bit more computational time. */
    for (int iter = 0; iter < iters; iter++)
    {
        /* The actual parallel loop. */
        for(int i = 0; i < len; i++) {
            for(int j = 0; j < len; j++)
            {
                float t = 0;
                for(int k = 0; k < len; k++)
                    t += a[k + j * len] * b[i + k * len];
                x[i + j *len] = t;
            }
        }
        {float *tmp = x; x = a; a = tmp; }
    }

    memcpy(x, a, len * len * sizeof(float));
}
#endif

#ifdef MATMUL_ACC
void matmul(float *a, float *b, float *x, int len, int iters)
{
    #pragma acc data copyin(a[0:len*len], b[0:len*len], iters, len) copyout(x[0:len*len])
    {
        for (int iter = 0; iter < iters; iter++) {
            #pragma acc parallel loop gang vector collapse(2)
            for(int i = 0; i < len; i++)
            {
                for(int j = 0; j < len; j++)
                {
                    float t = 0;
                    for(int k = 0; k < len; k++)
                        t += a[k + j * len] * b[i + k * len];
                    x[i + j * len] = t;
                }
            }

            //Copy x to a
            #pragma acc parallel loop gang vector
            for(int i = 0; i < len * len; i++) {
              a[i] = x[i];
            }
        }
    }
}
#endif

#ifdef MATMUL_ACC_ADV
void matmul(float *a, float *b, float *x, int len, int iters)
{
    size_t s = sizeof(float) * len * len;

    //Allocate memory on device
    float *d_a = acc_malloc(s);
    float *d_b = acc_malloc(s);
    float *d_x = acc_malloc(s);

    //Copy a and b to device
    acc_memcpy_to_device(d_a, a, s);
    acc_memcpy_to_device(d_b, b, s);

    #pragma acc data copyin(iters, len) deviceptr(d_a, d_b, d_x)
    {
        for (int iter = 0; iter < iters; iter++) {
            #pragma acc parallel loop gang collapse(2)
            for(int i = 0; i < len; i++)
            {
                for(int j = 0; j < len; j++)
                {
                    float t = 0;
                    #pragma acc loop vector reduction(+:t)
                    for(int k = 0; k < len; k++)
                        t += d_a[k + j * len] * d_b[i + k * len];
                    d_x[i + j * len] = t;
                }
            }

            { float *tmp = d_x; d_x = d_a; d_a = tmp; }
        }
        /* copy x back from device. */
        acc_memcpy_from_device(x, d_a, s);
    }
}
#endif

#ifdef MATMUL_OMP
void matmul(float *a, float *b, float *x, int len, int iters)
{
    #pragma omp parallel
    for (int iter = 0; iter < iters; iter++)
    {
        #pragma omp for 
        for(int i = 0; i < len; i++) {
            for(int j = 0; j < len; j++)
            {
                float t = 0;
                for(int k = 0; k < len; k++)
                    t += a[k + j * len] * b[i + k * len];
                x[i + j *len] = t;
            }
        }
        #pragma omp single
        {float *tmp = x; x = a; a = tmp; }
    }
}
#endif

void info()
{
#ifdef MATMUL_SEQ
    printf("Running matmul sequential\n");
#endif

#ifdef MATMUL_SSE
    printf("Running matmul using vectorization (sse)\n");
#endif

#ifdef MATMUL_ACC
    printf("Running matmul using OpenACC\n");
#endif

#ifdef MATMUL_ACC_ADV
    printf("Running matmul using OpenACC Advanced\n");
#endif

#ifdef MATMUL_OMP
#pragma omp parallel
    printf("Running matmul using OpenMP (Thread %d of %d threads)\n",
            omp_get_thread_num(),
            omp_get_num_threads());
#endif
}

void print_matrix(int n, float *m)
{
    printf("[ ");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            printf("% 8.02f ", m[i + j * n]);
        if (i < n-1)
            printf("\n  ");
    }
    printf(" ]\n");
}

void do_compute(int n, int iters)
{
    float *a = NULL, *b = NULL, *x = NULL;
    struct timeval before, after;

    size_t memsize = 3*n*n*sizeof(float);
    printf("Try to allocate %d * %d * %d * %zu = %zu bytes (%zu MB) of memory.\n", 3, n, n, sizeof(float), memsize, memsize/(1024*1024));

    a = malloc(n * n * sizeof(float));
    b = malloc(n * n * sizeof(float));
    x = malloc(n * n * sizeof(float));
    assert(a); assert(b); assert(x);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
        {
            a[i + j * n] = i + j;
            b[i + j * n] = i - j;
        }
    }

    if (debug) {
      print_matrix(n, a);
      print_matrix(n, b);
    }

    memset(x, 0, sizeof(float) * n * n);

    gettimeofday(&before, NULL);

    matmul(a, b, x, n, iters);

    gettimeofday(&after, NULL);

    double time = (double)(after.tv_sec - before.tv_sec) +
        (double)(after.tv_usec - before.tv_usec) / 1e6;

    printf("Computed in %f seconds!\n", time);

    if (debug) {
      print_matrix(n, x);
    }
}

int main(int argc, char **argv)
{

    int n = 100;
    int iters = 1;

    if(argc > 1) {
      n = atoi(argv[1]);
    }
    if(argc > 2) {
      iters = atoi(argv[2]);
    }
    if (argc > 3) {
      debug = atoi(argv[3]);
    }

    info();

    printf("Multiply %dx%d matrices.\n", n, n);

    #ifdef MATMUL_ACC
    acc_init(acc_device_nvidia);
    #endif

    do_compute(n, iters);

    return 0;

}

