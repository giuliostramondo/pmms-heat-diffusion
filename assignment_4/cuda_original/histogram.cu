#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Utility function, use to do error checking.

   Use this function like this:

   checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));

   And to check the result of a kernel invocation:

   checkCudaCall(cudaGetLastError());
*/
static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        printf("cuda Error \n");
	exit(1);
    }
}


__global__ void vectorAddKernel(float* deviceA, float* deviceB, float* deviceResult) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
// insert operation here
    deviceResult[i] = deviceA[i]+deviceB[i];
}

void vectorAddCuda(int n, float* a, float* b, float* result) {
    int threadBlockSize = 512;

    // allocate the vectors on the GPU
    float* deviceA = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceA, n * sizeof(float)));
    if (deviceA == NULL) {
        printf("could not allocate memory!\n");
        return;
    }
    float* deviceB = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceB, n * sizeof(float)));
    if (deviceB == NULL) {
        checkCudaCall(cudaFree(deviceA));
        printf("could not allocate memory!\n");
        return;
    }
    float* deviceResult = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceResult, n * sizeof(float)));
    if (deviceResult == NULL) {
        checkCudaCall(cudaFree(deviceA));
        checkCudaCall(cudaFree(deviceB));
        printf("could not allocate memory!\n");
        return;
    }


    // copy the original vectors to the GPU
    checkCudaCall(cudaMemcpy(deviceA, a, n*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(deviceB, b, n*sizeof(float), cudaMemcpyHostToDevice));

    // execute kernel
    vectorAddKernel<<<n/threadBlockSize, threadBlockSize>>>(deviceA, deviceB, deviceResult);
    cudaDeviceSynchronize();
    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    // copy result back
    checkCudaCall(cudaMemcpy(result, deviceResult, n * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaCall(cudaMemcpy(b, deviceB, n * sizeof(float), cudaMemcpyDeviceToHost));

    checkCudaCall(cudaFree(deviceA));
    checkCudaCall(cudaFree(deviceB));
    checkCudaCall(cudaFree(deviceResult));
}

int main(int argc, char* argv[]) {
    int n = 655360;
    float* a = new float[n];
    float* b = new float[n];
    float* result = new float[n];

    if (argc > 1) n = atoi(argv[1]);

    // initialize the vectors.
    for(int i=0; i<n; i++) {
        a[i] = i;
        b[i] = i;
    }

    vectorAddCuda(n, a, b, result);
    
    delete[] a;
    delete[] b;
    delete[] result;
    
    return 0;
}
