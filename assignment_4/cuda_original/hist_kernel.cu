#include <cuda.h>

__global__ void histogram(int *in, int* out) 
{
  int i=blockIdx.x*blockDim.x + threadIdx.x;
  out[i] = in[i];
} 
