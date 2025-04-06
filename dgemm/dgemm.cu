#include <stdio.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define BLOCK_SIZE 16

#define CUDA_CHECK(cmd) \
    do { \
        cudaError_t err = (cmd); \
        if (err != cudaSuccess) { \
            printf("Failed: Cuda error %s:%d '%s'\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define NCCL_CHECK(cmd) \
    do { \
        ncclResult_t res = (cmd); \
        if (res != ncclSuccess) { \
            printf("Failed: NCCL error %s:%d '%s'\n", \
                __FILE__, __LINE__, ncclGetErrorString(res)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void matmul(float* A, float* B, float* C, const int M, const int N, const int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 1d sharding
torch::Tensor distributed_matmul_v1(torch::Tensor A, torch::Tensor B, torch::Tensor C, const int M, const int N, const int K) {
    int nDevices;
    CUDA_CHECK(cudaGetDeviceCount(&nDevices));
    if (nDevices != 4) {
        throw std::runtime_error("This program requires exactly 4 GPUs");
    }

    ncclComm_t comms[4];
    float *A_d[4], *B_d[4];
    float *C_row[4];   
    float *C_full[4];  
    cudaStream_t streams[4];
    
    NCCL_CHECK(ncclCommInitAll(comms, nDevices, NULL));

    // Device memory allocation and data distribution
    for (int i = 0; i < nDevices; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        
        CUDA_CHECK(cudaMalloc(&A_d[i], (M/nDevices) * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&B_d[i], K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&C_row[i], (M/nDevices) * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&C_full[i], M * N * sizeof(float)));

        // Copy data from PyTorch tensors
        auto A_slice = A.slice(0, i*M/nDevices, (i+1)*M/nDevices);
        CUDA_CHECK(cudaMemcpyAsync(A_d[i], A_slice.data_ptr<float>(), 
                            (M/nDevices) * K * sizeof(float), 
                            cudaMemcpyDeviceToDevice, streams[i]));
        CUDA_CHECK(cudaMemcpyAsync(B_d[i], B.data_ptr<float>(), 
                            K * N * sizeof(float), 
                            cudaMemcpyDeviceToDevice, streams[i]));

    }

    // Launch kernels
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
            (M/nDevices + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (int i = 0; i < nDevices; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        matmul<<<gridDim, blockDim, 0, streams[i]>>>(
                A_d[i], B_d[i], C_row[i], M/nDevices, N, K);
    }

    // AllGather results
    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < nDevices; i++) {
        NCCL_CHECK(ncclAllGather(
            C_row[i], C_full[i], (M/nDevices)*N, ncclFloat,
            comms[i], streams[i]
        ));
    }
    NCCL_CHECK(ncclGroupEnd());

    // Copy result back to output tensor
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpyAsync(C.data_ptr<float>(), C_full[0], 
                         M * N * sizeof(float),
                         cudaMemcpyDeviceToDevice));

    // Cleanup
    for (int i = 0; i < nDevices; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaFree(A_d[i]));
        CUDA_CHECK(cudaFree(B_d[i]));
        CUDA_CHECK(cudaFree(C_row[i]));
        CUDA_CHECK(cudaFree(C_full[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        ncclCommDestroy(comms[i]);
    }

    return C;
}




// 2d sharding w/ overlap?
torch::Tensor distributed_matmul_v2(torch::Tensor A, torch::Tensor B, torch::Tensor C, const int M, const int N, const int K) {
    int nDevices = 4; // determines how many tiles per stage
    /*
    CUDA_CHECK(cudaGetDeviceCount(&nDevices));
    if (nDevices != 4) {
        throw std::runtime_error("This program requires exactly 4 GPUs");
    }
    */
    
    const int N_STAGES = 4; // how many stages A/B is partitioned
    const int M_TILE = M/N_STAGES;
    const int N_TILE = N/N_STAGES;
    const int K_TILE = K/N_STAGES;

    ncclComm_t comms[4];
    float *A_d[N_STAGES][nDevices]; // for A: stages along row_dim, GPU allocation along col_dim
    float *B_d[nDevices][N_STAGES]; // for B: stages along col_dim, GPU allocation along row_dim
    float *partial_C_d[nDevices][N_STAGES][N_STAGES]; // store results from each stage
    float *C_complete[nDevices][N_STAGES][N_STAGES]; // will contain the full output matrix (split into tiles)

    bool compute_graphCreated = false;
    cudaGraph_t compute_graph;
    cudaGraphExec_t instance;

    // spawn stream per device
    cudaStream_t compute_streams[nDevices];
    cudaStream_t communication_streams[nDevices];

    cudaEvent_t compute_done[nDevices][N_STAGES][N_STAGES]; // indicate when gemms are done

    NCCL_CHECK(ncclCommInitAll(comms, nDevices, NULL)); // create single process for multiple devices

    // setup streams and events
    for (int i = 0; i < nDevices; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamCreate(&compute_streams[i]));
        CUDA_CHECK(cudaStreamCreate(&communication_streams[i]));
        
        for (int row_stage = 0; row_stage < N_STAGES; row_stage++) {
            for (int col_stage = 0; col_stage < N_STAGES; col_stage++) {
                CUDA_CHECK(cudaEventCreate(&compute_done[i][row_stage][col_stage]));
            }
        }
    }

    // allocation/initialization
    for (int i = 0; i < nDevices; i++) {
        CUDA_CHECK(cudaSetDevice(i)); // host thread will now operate on the ith GPU
        for (int row_stage = 0; row_stage < N_STAGES; row_stage++) {
            CUDA_CHECK(cudaMalloc(&A_d[row_stage][i], M_TILE * K_TILE * sizeof(float)));
            for (int col_stage = 0; col_stage < N_STAGES; col_stage++) {
                // We allocate B for each column stage - B doesn't depend on row_stage
                if (row_stage == 0) {
                    CUDA_CHECK(cudaMalloc(&B_d[i][col_stage], K_TILE * N_TILE * sizeof(float)));
                }
                CUDA_CHECK(cudaMalloc(&partial_C_d[i][row_stage][col_stage], M_TILE * N_TILE * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&C_complete[i][row_stage][col_stage], M_TILE * N_TILE * sizeof(float)));
            }
        }
    }

    // computation stage
    for (int row_stage = 0; row_stage < N_STAGES; row_stage++) {
        int a_start_row = row_stage * M_TILE;
        int a_end_row = a_start_row + M_TILE;

        for (int col_stage = 0; col_stage < N_STAGES; col_stage++) {
            int b_start_col = col_stage * N_TILE;
            int b_end_col = b_start_col + N_TILE;
            
            for (int i = 0; i < nDevices; i++) {
                CUDA_CHECK(cudaSetDevice(i));
                
                // Each GPU loads its portion of A for this row_stage
                auto A_slice = A.slice(0, a_start_row, a_end_row).slice(1, i * K_TILE, (i+1) * K_TILE);
                A_slice = A_slice.contiguous();
                
                CUDA_CHECK(cudaMemcpyAsync(A_d[row_stage][i], 
                                          A_slice.data_ptr<float>(), 
                                          M_TILE * K_TILE * sizeof(float), 
                                          cudaMemcpyDeviceToDevice, 
                                          compute_streams[i]));
                
                // Each GPU loads its portion of B for this col_stage
                auto B_slice = B.slice(0, i * K_TILE, (i+1) * K_TILE).slice(1, b_start_col, b_end_col);
                B_slice = B_slice.contiguous();
                
                CUDA_CHECK(cudaMemcpyAsync(B_d[i][col_stage], 
                                          B_slice.data_ptr<float>(), 
                                          K_TILE * N_TILE * sizeof(float), 
                                          cudaMemcpyDeviceToDevice, 
                                          compute_streams[i]));
                
                // Launch kernel in compute stream
                dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
                dim3 gridDim((N_TILE + BLOCK_SIZE - 1) / BLOCK_SIZE, (M_TILE + BLOCK_SIZE - 1) / BLOCK_SIZE);
                matmul<<<gridDim, blockDim, 0, compute_streams[i]>>>(
                    A_d[row_stage][i], 
                    B_d[i][col_stage], 
                    partial_C_d[i][row_stage][col_stage], 
                    M_TILE, N_TILE, K_TILE);
               	CUDA_CHECK(cudaStreamSynchronize(compute_streams[i]));
                // create event to signal completion
                CUDA_CHECK(cudaEventRecord(compute_done[i][row_stage][col_stage], compute_streams[i]));
            }
        }
    }

    // communication stage
    for (int row_stage = 0; row_stage < N_STAGES; row_stage++) {
        for (int col_stage = 0; col_stage < N_STAGES; col_stage++) {
            for (int i = 0; i < nDevices; i++) {
                CUDA_CHECK(cudaSetDevice(i));
                CUDA_CHECK(cudaStreamSynchronize(compute_streams[i]));
                CUDA_CHECK(cudaStreamWaitEvent(communication_streams[i], 
                                            compute_done[i][row_stage][col_stage], 0));
            }
            NCCL_CHECK(ncclGroupStart());
            for (int i = 0; i < nDevices; i++) {
                void* src_buff = partial_C_d[i][row_stage][col_stage];
                void* recv_buff = C_complete[i][row_stage][col_stage];
                
                NCCL_CHECK(ncclAllReduce(src_buff, recv_buff, M_TILE*N_TILE, ncclFloat, ncclSum, 
                           comms[i], communication_streams[i]));
            }
            NCCL_CHECK(ncclGroupEnd());
        }
    }

    // wait for everyone to be done
    for (int i = 0; i < nDevices; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamSynchronize(compute_streams[i]));
        CUDA_CHECK(cudaStreamSynchronize(communication_streams[i]));
    }

    // Copy from C_complete on GPU 0 to output tensor C
    CUDA_CHECK(cudaSetDevice(0));

    // Allocate temporary buffer for the complete matrix
    float* C_temp;
    CUDA_CHECK(cudaMalloc(&C_temp, M * N * sizeof(float)));

    // Copy each tile to its correct position in C_temp
    for (int row_stage = 0; row_stage < N_STAGES; row_stage++) {
        for (int col_stage = 0; col_stage < N_STAGES; col_stage++) {
            // Calculate destination offset in the final matrix
            size_t row_offset = row_stage * M_TILE;
            size_t col_offset = col_stage * N_TILE;
            
            // Copy tile row by row to maintain proper stride
            for (int row = 0; row < M_TILE; row++) {
                size_t src_offset = row * N_TILE;
                size_t dst_offset = (row_offset + row) * N + col_offset;

                CUDA_CHECK(cudaMemcpy(C_temp + dst_offset, 
                                     C_complete[0][row_stage][col_stage] + src_offset, 
                                     N_TILE * sizeof(float), 
                                     cudaMemcpyDeviceToDevice));
            }
        }
    }
    CUDA_CHECK(cudaMemcpy(C.data_ptr<float>(), C_temp, M * N * sizeof(float), cudaMemcpyDeviceToDevice));

    // Free memory
    CUDA_CHECK(cudaFree(C_temp));
    for (int i = 0; i < nDevices; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamDestroy(compute_streams[i]));
        CUDA_CHECK(cudaStreamDestroy(communication_streams[i]));

        for (int row_stage = 0; row_stage < N_STAGES; row_stage++) {
            CUDA_CHECK(cudaFree(A_d[row_stage][i]));
            
            for (int col_stage = 0; col_stage < N_STAGES; col_stage++) {
                if (row_stage == 0) {  // Free B only once per column stage
                    CUDA_CHECK(cudaFree(B_d[i][col_stage]));
                }
                
                CUDA_CHECK(cudaFree(partial_C_d[i][row_stage][col_stage]));
                CUDA_CHECK(cudaFree(C_complete[i][row_stage][col_stage]));
                CUDA_CHECK(cudaEventDestroy(compute_done[i][row_stage][col_stage]));
            }
        }
        ncclCommDestroy(comms[i]);
    }

    return C;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("distributed_matmul_v1", &distributed_matmul_v1, "Distributed GEMM w/ no overlap");
    m.def("distributed_matmul_v2", &distributed_matmul_v2, "Distributed GEMM w/ sharding");
}
