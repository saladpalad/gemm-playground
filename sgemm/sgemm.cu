#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <cublas_v2.h>

__global__ void matmul_v1(float* A, float* B, float* C, const int m, const int n, const int d){
    // A is (m,d), B is (d,n), C is (m,n)

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    if(i < m && j < n){
        float inner_prod = 0.0f;
        for(int k = 0; k < d; k++){
            //printf("Thread: (%d, %d), Loading from A: %f\n", threadIdx.x, threadIdx.y, A[i*d+k]);
            inner_prod += A[i*d+k] * B[k*n+j];
        }
        C[i*n+j] = inner_prod;
    }
}

__global__ void matmul_v2(float* A, float* B, float* C, const int m, const int n, const int d){
    const int num_threads = 16;

    //int i = blockIdx.x*num_threads + (threadIdx.x / num_threads);
    //int j = blockIdx.y*num_threads + (threadIdx.x % num_threads);
    
    // This will also be coalesced
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;

    if(i < m && j < n){
        float inner_prod = 0.0f;
        for(int k = 0; k < d; k++){
            inner_prod += A[i*d+k] * B[k*n+j];
        }
        C[i*n+j] = inner_prod;
    }
}

__global__ void matmul_v3(float* A, float* B, float* C, const int m, const int n, const int d){
    // A.T is (d,m), B is (d,n), C is (m,n)
    // make A transpose for Coalesced

    float C_local[8][8];
    for (int ib = 0; ib < 8; ++ib) {
        for (int jb = 0; jb < 8; ++jb) {
            C_local[ib][jb] = 0;
        }
    }

    for (int k = 0; k < d; ++k) {
        float x[8];
        float y[8];
        for (int ib = 0; ib < 8; ++ib) {
            int i = blockIdx.y * 64 + ib * 8 + threadIdx.y;
            x[ib] = A[i*d + k];
        }
        for (int jb = 0; jb < 8; ++jb) {
            int j = blockIdx.x * 64 + jb * 8 + threadIdx.x;
            y[jb] = B[k*n + j];
        }
        for (int ib = 0; ib < 8; ++ib) {
            for (int jb = 0; jb < 8; ++jb) {
                C_local[ib][jb] += x[ib] * y[jb];
            }
        }
    }

    for (int ib = 0; ib < 8; ++ib) {
        for (int jb = 0; jb < 8; ++jb) {
            int i = blockIdx.y * 64 + ib * 8 + threadIdx.y;
            int j = blockIdx.x * 64 + jb * 8 + threadIdx.x;
            if (i < m && j < n) {
                C[i*n + j] = C_local[ib][jb];
            }
        }
    }
}

__global__ void matmul_v4(float* A, float* B, float* C, const int m, const int n, const int d){
    // A is (m,d), B is (d,n), C is (m,n)
       
    __shared__ float As[8][64];
    __shared__ float Bs[8][64];

    float C_local[8][8];
    for (int ib = 0; ib < 8; ++ib) {
        for (int jb = 0; jb < 8; ++jb) {
            C_local[ib][jb] = 0;
        }
    }

    for (int k = 0; k < d; k+=8) {
        // each thread will load 8 elements into SRAM
        int local_thread = threadIdx.y * blockDim.x + threadIdx.x; //0-63 threads
        int i = blockIdx.x * 64 + local_thread;
        int j = blockIdx.y * 64 + local_thread;

        for(int f = 0; f < 8; f++){
            int local_k = k + f;
            As[f][local_thread] = A[local_k*m + i];
            Bs[f][local_thread] = B[local_k*n + j];
        }
        __syncthreads();
        
        #pragma unroll
        for(int f = 0; f < 8; f++){
            float x[8];
            float y[8];
            for (int ib = 0; ib < 8; ++ib) {
                x[ib] = As[f][ib*8+threadIdx.x];
                for (int jb = 0; jb < 8; ++jb) {
                    y[jb] = Bs[f][jb*8+threadIdx.y];
                    //reuse x[ib] and y[jb]
                    C_local[ib][jb] += x[ib] * y[jb];
                }
            }
        }
        __syncthreads();
    }

    for (int ib = 0; ib < 8; ++ib) {
        for (int jb = 0; jb < 8; ++jb) {
            int i = blockIdx.x * 64 + ib * 8 + threadIdx.x;
            int j = blockIdx.y * 64 + jb * 8 + threadIdx.y;
            if (i < m && j < n) {
                C[i*n + j] = C_local[ib][jb];
            }
        }
    }
}

torch::Tensor launch_matmul_v1(torch::Tensor A, torch::Tensor B, torch::Tensor C, const int m, const int n, const int d){
    const int num_threads = 16;
    dim3 blockDim(num_threads, num_threads); // 256 threads per block

    const int grid_size_x = (m+num_threads-1)/num_threads; // spawn enough blocks to process m
    const int grid_size_y = (n+num_threads-1)/num_threads; // spawn enough blocks to process n
    dim3 gridDim(grid_size_x, grid_size_y);

    matmul_v1<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        m, n, d
    );
    return C;
}

torch::Tensor launch_matmul_v2(torch::Tensor A, torch::Tensor B, torch::Tensor C, const int m, const int n, const int d){
    const int num_threads = 16;
    dim3 blockDim(num_threads * num_threads); // 256 threads per block

    const int grid_size_x = (m+num_threads-1)/num_threads; // spawn enough blocks to process m
    const int grid_size_y = (n+num_threads-1)/num_threads; // spawn enough blocks to process n
    dim3 gridDim(grid_size_x, grid_size_y);

    matmul_v2<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        m, n, d);
    return C;
}

torch::Tensor launch_matmul_v3(torch::Tensor A, torch::Tensor B, torch::Tensor C, const int m, const int n, const int d){
    // thread -> processes 8x8 output, block has 8x8 threads so 1 block -> processes 64x64 output elements

    const int num_threads = 8;
    dim3 blockDim(num_threads, num_threads);

    const int grid_size_x = (m+63)/64;
    const int grid_size_y = (n+63)/64;
    dim3 gridDim(grid_size_x, grid_size_y);

    matmul_v3<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        m, n, d
    );
    return C;
}

torch::Tensor launch_matmul_v4(torch::Tensor A, torch::Tensor B, torch::Tensor C, const int m, const int n, const int d){
    // thread -> loads 8 elements from DRAM to shared (8*64 total elements), then load 8+8 elements into registers
    // thread -> processes 8x8 output, block has 8x8 threads so 1 block -> processes 64x64 output elements

    const int num_threads = 8;
    dim3 blockDim(num_threads, num_threads);

    const int grid_size_x = (m+63)/64;
    const int grid_size_y = (n+63)/64;
    dim3 gridDim(grid_size_x, grid_size_y);

    matmul_v4<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        m, n, d
    );
    return C;
}

torch::Tensor launch_matmul_cublas(torch::Tensor A, torch::Tensor B, torch::Tensor C, const int m, const int n, const int k){
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // c = alpha*A * B + beta*C
    // row major: (mxk) * (kxn) = mxn
    // col major: (kxm) * (nxk) so do B*A instead for (nxk) * (kxm) = nxm
    // nxm col major becomes mxn in row major order
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B.data_ptr<float>(), n, A.data_ptr<float>(), k, &beta, C.data_ptr<float>(), n);
    cublasDestroy(handle);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_v1", &launch_matmul_v1, "Naive Matmul");
  m.def("matmul_v2", &launch_matmul_v2, "Coalesced Matmul");
  m.def("matmul_v3", &launch_matmul_v3, "Register Reuse Matmul");
  m.def("matmul_v4", &launch_matmul_v4, "Shared Mem Block Matmul");
  m.def("matmul_cublas", &launch_matmul_cublas, "CUBLAS SGEMM");
}

