#include <stdlib.h>
#include <stdio.h>
#include <openacc.h>

/* Addition function to offload to GPU */
void vecadd(float *restrict x, float *a, float *b, int len, int repeat){

  /* 
   * OpenACC region. 
   *
   * a and b are inputs only, so only copy to device and not back.
   * x is output only, so only copy from device after exec.
   * */
  #pragma acc parallel copyin(a[0:len], b[0:len]) copyout(x[0:len])
  {
    /* Loop such that it may consume a bit more computational time. */
    for (int j = 0; j < repeat; j++) {

      /* The actual parallel loop. */
      #pragma acc loop gang worker
      for(int i = 0; i < len; i++){

        /* Add elements */
        x[i] = a[i] + b[i];
      }
    }
  }
}

/*
 * Function that gets some OpenACC statistics, allocated memory on host,
 * and calls compute kernel. 
 */
void do_compute(size_t n, size_t m)
{

  /* See which devices we have */
  int ndev = acc_get_num_devices(acc_device_nvidia);
  printf("Num NVIDIA (%d): %d\n", acc_device_nvidia, ndev);

  int rdev = acc_get_num_devices(acc_device_radeon);
  printf("Num RADEON (%d): %d\n", acc_device_radeon, rdev);

  int xdev = acc_get_num_devices(acc_device_xeonphi);
  printf("Num XEON   (%d): %d\n", acc_device_xeonphi, xdev);

  struct timeval before, after; 
 
  /* Calculate the amount of memory we try to allocate */
  size_t memsize = 3*n*sizeof(float);
  printf("Try to allocate %llu * %llu * %llu = %llu bytes (%llu MB) of memory.\n", 3, n, sizeof(float), memsize, memsize/(1024*1024));

  /* Some values that might be of interest? */
  printf("sizeof long: %llu, sizeof size_t: %llu, sizeof float: %llu, sizeof float*: %llu\n", sizeof(long), sizeof(size_t), sizeof(float), sizeof(float*));

  /* Allocate memory */
  float *a = (float*)malloc(n * sizeof(float));
  float *b = (float*)malloc(n * sizeof(float));
  float *x = (float*)malloc(n * sizeof(float));
  
  /* Did we have sufficient memory? */
  if(a == NULL || b == NULL || x == NULL){
    printf("Memory Alloc error...\n");
    exit(-1);
  }

  /* Initialize arrays. */
  printf("\nInitialize...\n");
  for(int i = 0; i < n; i++){
    a[i] = i;
    b[i] = i;
    x[i] = 0;
  }

  /* Now we're ready to compute! */
  printf("Ready to compute...\n");

  gettimeofday(&before, NULL); 
 
  vecadd(x, a, b, n, m);

  gettimeofday(&after, NULL); 

  double time = (double)(after.tv_sec - before.tv_sec) +  
        (double)(after.tv_usec - before.tv_usec) / 1e6; 
 
  printf("Computed in %f seconds!\n", time);

  /* Check result for small arrays */
  if(n < 50){
    for(int i = 0; i < n; i++){
      printf("%f ", x[i]);
    }
    printf("\n");
  }


  /* Cleanup */
  free(a);
  free(b);
  free(x);

  printf("Done!\n");

}

int main(int argc, char **argv){

  /* Get number of additions + number of repetitions. */
  if(argc < 2){
    printf("Usage: ./%s size [repeat]\n", argv[0]);
    exit(-1);
  }

  size_t n = atol(argv[1]);
  size_t m = 1;

  if(argc > 2) m = atol(argv[2]);

  /* Conmpute */
  do_compute(n, m);

  return 0;
}

