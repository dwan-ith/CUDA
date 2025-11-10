#include <cuda_runtime.h>
#include <stdio.h>

__global__ void copyDataCoalesced(float *in, float *out, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        out[index] = in[index];
    }
}

int main() {
    int n = 1 << 20;
    size_t size = n * sizeof(float);

    float *h_in = (float*)malloc(size);
    float *h_out = (float*)malloc(size);

    for (int i = 0; i < n; i++) {
        h_in[i] = i;
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    copyDataCoalesced<<<numBlocks, blockSize>>>(d_in, d_out, n);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}
