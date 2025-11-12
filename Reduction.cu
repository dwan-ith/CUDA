#include <cuda_runtime.h>
#include <stdio.h>

__global__ void reduceSum(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : 0;
    if (i + blockDim.x < n)
        sdata[tid] += input[i + blockDim.x];
    
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

int main() {
    int n = 1 << 20;
    size_t bytes = n * sizeof(float);
    
    float *h_input = (float*)malloc(bytes);
    float *h_output;
    
    for (int i = 0; i < n; i++) {
        h_input[i] = 1.0f;
    }
    
    float *d_input, *d_output;
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);
    
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, blocksPerGrid * sizeof(float));
    
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    reduceSum<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_input, d_output, n);
    
    h_output = (float*)malloc(blocksPerGrid * sizeof(float));
    cudaMemcpy(h_output, d_output, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    
    float finalSum = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        finalSum += h_output[i];
    }
    
    printf("Sum: %f\n", finalSum);
    printf("Expected: %d\n", n);
    
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
