#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <time.h>

using std::cout;
using std::generate;
using std::vector;

void verify_result(vector<int> &a, vector<int> &b, vector<int> &c, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n ; j++) {
      int tmp = 0;
      for (int k = 0; k < n; k++) {
        tmp += a[i * n + k] * b[k * n + j];
      }

      // Check against the CPU result
      if (tmp != c[i*n+j]) {
        std::cerr << "Verification failed at (" << i << "," << j << "): " << tmp << " != " << c[i*n+j] << std::endl;
        std::cerr << "Matrix A row: ";
        for (int k = 0; k < n; k++) {
          std::cerr << a[i * n + k] << " ";
        }
        std::cerr << "\nMatrix B column: ";
        for (int k = 0; k < n; k++) {
            std::cerr << b[k * n + j] << " ";
            }
        std::cerr << "Mismatch at (" << i << "," << j << "): " << tmp << " != " << c[i*n+j] << std::endl;
        exit(1);
      }
    }
  }
}

#define SMEM_SIZE 1024 
__global__ void mm(int *a, int *b, int *c, int n) {
    
    __shared__ int tile_a[SMEM_SIZE];
    __shared__ int tile_b[SMEM_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int temp = 0;
    for (int tile = 0; tile < n; tile += blockDim.x) {
        tile_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * n + tile + threadIdx.x];
        tile_b[threadIdx.y * blockDim.x + threadIdx.x] = b[tile *n + threadIdx.y * n  + col];
        __syncthreads();

    for ( int k = 0; k < blockDim.x; k++) {
        temp += tile_a[threadIdx.y* blockDim.x + k] * tile_b[k * blockDim.y + threadIdx.x];
    }
    __syncthreads();
    }

    c[row * n + col ] = temp;



}

__global__ void mm_transpose(int *a, int *b, int *c, int n) {
    
    __shared__ int tile_a[SMEM_SIZE];
    __shared__ int tile_b[SMEM_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int temp = 0;
    for (int tile = 0; tile < n; tile += blockDim.x) {
        tile_a[threadIdx.y * blockDim.x + threadIdx.x] = a[tile * n + threadIdx.x *n + row];
        tile_b[threadIdx.y * blockDim.x + threadIdx.x] = b[tile *n + threadIdx.y * n  + col];
        __syncthreads();

    for ( int k = 0; k < blockDim.x; k++) {
        temp += tile_a[threadIdx.y * blockDim.x + k] * tile_b[k * blockDim.y + threadIdx.x];
    }
    __syncthreads();
    }

    c[row * n + col ] = temp;



}

void transpose(vector<int> &in, vector<int> &out, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            out[j*n + i] = in[i*n + j];
        }
    }
}

void compare_matrices(vector<int> &a, vector<int> &b, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (a[i*n + j] != b[i*n + j]) {
                std::cerr << "Mismatch at (" << i << "," << j << "): " << a[i*n + j] << " != " << b[i*n + j] << std::endl;
                exit(1);
            }
        }
    }
    std::cout << "Matrices match!" << std::endl;
}

int main (int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " [matrix_size]" << std::endl;
        return 1;
    }
    int N = atoi(argv[1]);
    size_t bytes = N * N * sizeof(int);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  // Host vectors
    vector<int> h_a(N * N);
    vector<int> h_a_transposed(N * N);
    vector<int> h_b(N * N);
    vector<int> h_c(N * N);
    vector<int> h_c_transposed(N * N);

  // Initialize matrices
    generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
    generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });
    
    transpose(h_a, h_a_transposed, N);


    int * d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 THREADS(32,32);
    dim3 BLOCKS(N/THREADS.x, N/THREADS.y);

    // Launch the kernel
    cudaEventRecord(start);
    mm<<<BLOCKS, THREADS>>>(d_a,d_b, d_c, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
       float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "tiled Kernel execution time: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(d_c,0, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, h_a_transposed.data(), bytes, cudaMemcpyHostToDevice);
    // Launch the kernel with transposed matrix
    cudaEventRecord(start); 
    mm_transpose<<<BLOCKS, THREADS>>>(d_a,d_b, d_c, N);
    cudaEventRecord(stop);  
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "tiled with transposed A Kernel execution time: " << milliseconds << " ms" << std::endl;
    cudaMemcpy(h_c_transposed.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    compare_matrices(h_c, h_c_transposed, N);

    clock_t cpu_start = clock();
    verify_result(h_a, h_b, h_c, N);
    clock_t cpu_end = clock();
    double cpu_time = double(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000;
    std::cout << "CPU verification time: " << cpu_time << " ms" << std::endl;

    std::cout << "Passed!" << std::endl;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}


