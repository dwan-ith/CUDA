#include <stdio.h>

#define BLOCK_SIZE 1024

__global__ void scan_phase1(float *out, float *in, float *block_sums, int n)
{
    extern __shared__ float s[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    s[tid] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride <<= 1)
    {
        float val = (tid >= stride) ? s[tid - stride] : 0.0f;
        __syncthreads();
        if (tid >= stride) s[tid] += val;
        __syncthreads();
    }

    if (block_sums && tid == blockDim.x - 1)
        block_sums[blockIdx.x] = s[tid];

    if (tid == blockDim.x - 1) s[tid] = 0.0f;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (tid >= stride)
        {
            float t = s[tid - stride];
            s[tid - stride] = s[tid];
            s[tid] += t;
        }
        __syncthreads();
    }

    if (idx < n) out[idx] = s[tid];
}

__global__ void add_block_sums(float *out, float *block_sums, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x + blockDim.x;
    if (idx < n)
        out[idx] += block_sums[blockIdx.x];
}

int main()
{
    int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float *h_in = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);

    for (int i = 0; i < N; i++) h_in[i] = 1.0f;

    float *d_in, *d_out, *d_block_sums;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMalloc(&d_block_sums, ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) * sizeof(float));

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    scan_phase1<<<grid, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_out, d_in, d_block_sums, N);
    scan_phase1<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_block_sums, d_block_sums, nullptr, grid);

    add_block_sums<<<grid, BLOCK_SIZE>>>(d_out, d_block_sums, N);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    printf("Scan complete. First 10 values: ");
    for (int i = 0; i < 10; i++) printf("%.0f ", h_out[i]);
    printf("\n");

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_block_sums);
    free(h_in); free(h_out);
    return 0;
}