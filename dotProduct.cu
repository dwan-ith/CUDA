#define THREADS_PER_BLOCK 256

__global__ void dot_product_kernel(float* a, float* b, float* partial_sum, int N) {
    __shared__ float cache[THREADS_PER_BLOCK];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    float temp = 0;
    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    cache[cacheIndex] = temp;
    __syncthreads();
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }
    if (cacheIndex == 0) partial_sum[blockIdx.x] = cache[0];
}

void dot_product(float* h_a, float* h_b, float* result, int N) {
    float *d_a, *d_b, *d_partial_sum;
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_partial_sum, blocks * sizeof(float));
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
    dot_product_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_partial_sum, N);
    float* h_partial_sum = (float*)malloc(blocks * sizeof(float));
    cudaMemcpy(h_partial_sum, d_partial_sum, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    float sum = 0;
    for (int i = 0; i < blocks; i++) sum += h_partial_sum[i];
    *result = sum;
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_partial_sum); free(h_partial_sum);
}
