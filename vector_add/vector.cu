#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " " << file << ":" << line << std::endl;
        exit(code);
    }
}

void vec_init(int*v, int size) {
    for (int i = 0; i < size; i++) {
        v[i] = rand() % 100;
    }
}

__global__ void vec_add(int *a, int * b, int*c, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    int size = 1024 * 1024;
    size*= 1024   ;

    int *h_a, *h_b, *h_c;
    int * d_a, *d_b, *d_c;
    h_a = (int*)malloc(size * sizeof(int));
    h_b = (int*)malloc(size * sizeof(int));
    h_c = (int*)malloc(size * sizeof(int));

    vec_init(h_a, size);
    vec_init(h_b, size);
    cudaCheckError(cudaMalloc(&d_a, size * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_b, size * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_c, size * sizeof(int)));
    cudaCheckError(cudaMemcpy(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_b, h_b, size * sizeof(int), cudaMemcpyHostToDevice));
    dim3 blocksize(1024);
    int GridSize = ceil((float)size / blocksize.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vec_add<<<GridSize, blocksize>>>(d_a, d_b, d_c, size);
    cudaCheckError(cudaGetLastError());
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaCheckError(cudaMemcpy(h_c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost));
 
    // CPU vector addition with time measurement
    int *h_c_cpu = (int*)malloc(size * sizeof(int));
    clock_t cpu_start = clock();
    for (int i = 0; i < size; i++) {
        h_c_cpu[i] = h_a[i] + h_b[i];
    }
    clock_t cpu_end = clock();
    double cpu_milliseconds = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
    std::cout << "CPU execution time: " << cpu_milliseconds << " ms" << std::endl;
    free(h_c_cpu);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}
