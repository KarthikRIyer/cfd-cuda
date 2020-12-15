//
// Created by karthik on 15/12/20.
//

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include <chrono>

// CUDA kernel for vector addition
// __global__ mean this is called from CPU and is run on GPU
__global__ void vectorAdd(const int *__restrict a, const int *__restrict b, int *__restrict c, int N) {
    // calculate global thread id
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < N) c[tid] = a[tid] + b[tid];
}

void verify_result(std::vector<int> &a, std::vector<int> &b, std::vector<int> &c) {
    for (int i = 0; i < a.size(); i++) {
        assert(c[i] == a[i] + b[i]);
    }
}

int main() {
    constexpr int N = 1 << 16;
    constexpr size_t bytes = sizeof(int) * N;

    // vector for holding CPU side data
    std::vector<int> a;
    a.reserve(N);
    std::vector<int> b;
    b.reserve(N);
    std::vector<int> c;
    c.reserve(N);

    for (int i = 0; i < N; i++) {
        a.push_back(rand() % 100);
        b.push_back(rand() % 100);
    }

    auto start = std::chrono::system_clock::now();

    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Serial addition : " << elapsed_seconds.count() << "s \n";
    c.clear();
    c.reserve(N);

    // allocate memory on device
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // copy data from host to device (cpu to gpu)
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    // threads per CTA (1024)
    int NUM_THREADS = 1 << 10;

    // CTAs per grid
    // we need to launch at least as many threads as we have elements
    // adds extra CTA to the grid if N cannot be evenly divided by NUM_THREADS
    int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

    start = std::chrono::system_clock::now();
    // launch kernel on GPU
    // asynchronous
    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);

    // copy sum vector from device to host
    // synchronous: waits for prior kernel launch to complete
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    end = std::chrono::system_clock::now();

    elapsed_seconds = end - start;
    std::cout << "Parallel addition : " << elapsed_seconds.count() << "s \n";

    verify_result(a, b, c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout << "COMPLETED SUCCESSFULLY !";

    return 0;
}