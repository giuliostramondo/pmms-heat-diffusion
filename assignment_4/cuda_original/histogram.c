#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <cuda.h>

static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        printf("cuda error \n");
        exit(1);
    }
}


void histogram(int *v, long n){
  int* deviceIn, *deviceOut;
  int threadBlockSize=256;
    
  checkCudaCall(cudaMalloc((void **) &deviceIn, n * sizeof(int)));
    if (deviceIn == NULL) {
        printf("Error in cudaMalloc! \n");
        return;
    }
  checkCudaCall(cudaMalloc((void **) &deviceOut, n * sizeof(int)));
    if (deviceOut == NULL) {
        checkCudaCall(cudaFree(deviceIn));
        printf("Error in cudaMalloc! \n");
        return;
    }


    checkCudaCall(cudaMemcpy(deviceIn, result, n * sizeof(int), cudaMemcpyDeviceToHost));
    vectorAddKernel<<<n/threadBlockSize, threadBlockSize>>>(deviceIn, deviceOut);
    cudaDeviceSynchronize();
    checkCudaCall(cudaMemcpy(result, deviceOut, n * sizeof(int), cudaMemcpyDeviceToHost));

    checkCudaCall(cudaFree(deviceIn));
    checkCudaCall(cudaFree(deviceOut));

}

int main(int argc, char **argv) {
  int length = 1024;
  srand(seed);

  /* Allocate vector. */
  vector = (int*)malloc(length*sizeof(int));
  if(vector == NULL) {
    fprintf(stderr, "Malloc failed...\n");
    return -1;
  }

  /* Fill vector. */
  for(long i = 0; i < length; i++) {
        vector[i] = (int)i;
  }

  histogram(vector, length);

  free(vector);

  return 0;
}

