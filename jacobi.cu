#include "jacobi.h"
#include <iostream>

#define MASK_DIM 3
#define MASK_OFFSET (MASK_DIM/2)

__constant__ double mask[MASK_DIM * MASK_DIM];

/*
__global__ void jacobikernel(double *psi_d, double *psinew_d, int m, int n, int numiter) {

    // calculate each thread's global row and col
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row > 0 && row <= m && col > 0 && col <= n) {
//        for (int i = 1; i <= numiter; i++) {
//        d_error = 0;
        psinew_d[row * (m + 2) + col] =
                0.25f * (psi_d[(row - 1) * (m + 2) + col] + psi_d[(row + 1) * (m + 2) + col] +
                         psi_d[(row) * (m + 2) + col - 1] + psi_d[(row) * (m + 2) + col + 1]);

//            __syncthreads();
//            psi_d[row * (m + 2) + col] = psinew_d[row * (m + 2) + col];
//            __syncthreads();
//        }
    }
}
 */

__global__ void convolution_2d(double *matrix, double *result, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int start_r = row - MASK_OFFSET;
    int start_c = col - MASK_OFFSET;

    extern __shared__ double s_matrix[];

    double temp = 0;

    if (row < N + 2 && col < N + 2) {
        s_matrix[threadIdx.y * blockDim.x + threadIdx.x] = matrix[row * (N + 2) + col];
    }
    __syncthreads();
    if (row > 0 && row <= N && col > 0 && col <= N) {
        for (int i = 0; i < MASK_DIM; i++) {
            for (int j = 0; j < MASK_DIM; j++) {

                if (start_c + j <= N + 1 && start_r + i <= N + 1) {
                    if (threadIdx.y + i >= blockDim.y || threadIdx.x + j >= blockDim.x) {
                        temp += matrix[(start_r + i) * (N + 2) + (start_c + j)] * mask[i * MASK_DIM + j];
                    } else {
                        temp += s_matrix[(threadIdx.y - MASK_OFFSET + i) * blockDim.x +
                                         (threadIdx.x - MASK_OFFSET + j)] * mask[i * MASK_DIM + j];
                    }
                }

            }
        }
        result[row * (N + 2) + col] = temp;
    }
}

void jacobiiter_gpu(double *psi, int m, int n, int numiter, double &error) {

    double *psi_d;
    double *psinew_d;
    size_t bytes = sizeof(double) * (m + 2) * (n + 2);
    size_t bytes_m = sizeof(double) * 3 * 3;

    //define mask
    double *h_mask = new double[3 * 3];
    h_mask[0] = 0;
    h_mask[1] = 0.25;
    h_mask[2] = 0;
    h_mask[3] = 0.25;
    h_mask[4] = 0;
    h_mask[5] = 0.25;
    h_mask[6] = 0;
    h_mask[7] = 0.25;
    h_mask[8] = 0;
    cudaMemcpyToSymbol(mask, h_mask, bytes_m);

    // allocate memory on gpu
    cudaMalloc(&psi_d, bytes);
    cudaMalloc(&psinew_d, bytes);

    // copy data to gpu
    cudaMemcpy(psi_d, psi, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(psinew_d, psi_d, bytes, cudaMemcpyDeviceToDevice);

    int THREADS = 16;
    int BLOCKS = (m + 2 + THREADS - 1) / THREADS;

    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    size_t SHMEM = (THREADS + 2) * (THREADS + 2) * sizeof(double);

    for (int i = 1; i <= numiter; i++) {
//        jacobikernel<<<blocks, threads>>>(psi_d, psinew_d, m, n, numiter);
        convolution_2d<<<blocks, threads, SHMEM>>>(psi_d, psinew_d, m);
        cudaMemcpy(psi_d, psinew_d, bytes, cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(psi, psi_d, bytes, cudaMemcpyDeviceToHost);

    cudaFree(psi_d);
    cudaFree(psinew_d);
    delete[] h_mask;
}

// serial cpu
void jacobistep(double *psinew, double *psi, int m, int n) {
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= m; j++) {
            psinew[i * (m + 2) + j] = 0.25f * (psi[(i - 1) * (m + 2) + j] + psi[(i + 1) * (m + 2) + j] +
                                               psi[(i) * (m + 2) + j - 1] + psi[(i) * (m + 2) + j + 1]);
        }
    }
}

// serial cpu
double deltasq(double *newarr, double *oldarr, int m, int n) {
    double dsq = 0;
    double tmp;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= m; j++) {
            tmp = newarr[i * (m + 2) + j] - oldarr[i * (m + 2) + j];
            dsq += tmp * tmp;
        }
    }

    return dsq;
}