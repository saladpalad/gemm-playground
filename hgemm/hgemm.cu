#include <mma.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include "ptx.h"


__launch_bounds__(16 * 16)
__global__ void matmul_v1(const half *A, const half *B, float *C, int M, int N, int K) {
  __shared__ uint4 As[32][8];
  __shared__ uint4 Bs[32][8];
  int mBlock = blockIdx.y*64;
  int nBlock = blockIdx.x*64;
  const uint4 *globalTileA = reinterpret_cast<const uint4 *>(A + mBlock * K);
  const uint4 *globalTileB = reinterpret_cast<const uint4 *>(B + nBlock * K);

  // warp layout is 2 x 4
  // (warp_0 | warp_1 | warp_2 | warp_3)
  // (warp_4 | warp_5 | warp_6 | warp_7)
  int threadID = threadIdx.y * blockDim.x + threadIdx.x;
  int warpID = threadID / 32;
  int laneID = threadID % 32;
  int mWarp = 16 * (warpID / 4);
  int nWarp = 8 * (warpID % 4);

  unsigned aReg[4];
  unsigned bReg[2];
  float dReg[2][2][4] = {0.};

  // row / column indices when storing to shared memory
  int storeRow = warpID * 4 + laneID / 8;
  int storeCol = (laneID % 8);

  // row/column indices when loading from permuted shmem layout to registers
  int loadRowA = (laneID % 16) / 2;
  int loadColA = (laneID / 16 + 4 * (laneID % 2));
  int loadRowB = (laneID % 8) / 2;
  int loadColB = (laneID / 8 + 4 * (laneID % 2));

  for (int k = 0; k < K/8; k += 4) {
    As[storeRow][storeCol] = globalTileA[(warpID*8 + laneID/4)*K/8 + k + laneID%4];
    Bs[storeRow][storeCol] = globalTileB[(warpID*8 + laneID/4)*K/8 + k + laneID%4];
    __syncthreads();

    // loop over the two (M/N=16, K=4) tiles of a and b
    for (int m = 0; m < 2; m++) {
      int mTile = m * 8;
      for (int n = 0; n < 2; n++) {
        int nTile = n * 4;
        load_matrix_x4(aReg, (As[mWarp + mTile + loadRowA] + loadColA));
        load_matrix_x2(bReg, (Bs[nWarp + nTile + loadRowB] + loadColB));
        mma_m16n8k16(aReg, bReg, dReg[m][n], dReg[m][n]);
        load_matrix_x4(aReg, (As[mWarp + mTile + loadRowA] + (loadColA^2)));
        load_matrix_x2(bReg, (Bs[nWarp + nTile + loadRowB] + (loadColB^2)));
        mma_m16n8k16(aReg, bReg, dReg[m][n], dReg[m][n]);
      }
    }
    __syncthreads();
  }

  int groupID     = laneID >> 2;
  int groupLaneID = laneID % 4;
  for (int m = 0; m < 2; m++) {
    for (int n = 0; n <  2; n++) {
      int mTile = m * 16;
      int nTile = n * 8;
      float2 d0 = make_float2(dReg[m][n][0], dReg[m][n][1]);
      float2 d2 = make_float2(dReg[m][n][2], dReg[m][n][3]);
      float2 *cOut0 = reinterpret_cast<float2 *>(C + (mBlock + mTile + 2*mWarp + groupID    )*N + nBlock + nTile + 2*nWarp + 2 * groupLaneID);
      float2 *cOut2 = reinterpret_cast<float2 *>(C + (mBlock + mTile + 2*mWarp + groupID + 8)*N + nBlock + nTile + 2*nWarp + 2 * groupLaneID);
      *cOut0 = d0;
      *cOut2 = d2;
    }
  }
  __syncthreads();
}

__launch_bounds__(16 * 16)
__global__ void matmul_v2(const half *A, const half *B, float *C, int M, int N, int K) {
  __shared__ uint4 As[32][8];
  __shared__ uint4 Bs[32][8];
  int mBlock = blockIdx.y*64;
  int nBlock = blockIdx.x*64;
  const uint4 *globalTileA = reinterpret_cast<const uint4 *>(A + mBlock * K);
  const uint4 *globalTileB = reinterpret_cast<const uint4 *>(B + nBlock * K);

  int threadID = threadIdx.y * blockDim.x + threadIdx.x;
  int warpID = threadID / 32;
  int laneID = threadID % 32;
  int mWarp = 16 * (warpID / 4);
  int nWarp = 8 * (warpID % 4);

  unsigned aReg[4]; // holds 8 fp16 values per thread
  unsigned bReg[2]; // holds 4 fp16 values per thread
  float dReg[2][2][4] = {0.};

  int storeRow = warpID * 4 + laneID / 8;
  // HINT: The code can be found in one of the readings/references ;)
  int storeCol = laneID % 8 ^ (laneID / 8);

  int loadRowA = (laneID % 16) / 2;
  int loadColA = (laneID / 16 + 4 * (laneID % 2)) ^ (loadRowA % 4);
  int loadRowB = (laneID % 8) / 2;
  int loadColB = (laneID / 8 + 4 * (laneID % 2)) ^ (loadRowB % 4);

  for (int k = 0; k < K/8; k += 4) {
    As[storeRow][storeCol] = globalTileA[(warpID*8 + laneID/4)*K/8 + k + laneID%4];
    Bs[storeRow][storeCol] = globalTileB[(warpID*8 + laneID/4)*K/8 + k + laneID%4];
    __syncthreads();

    for (int m = 0; m < 2; m++) {
      int mTile = m * 8;
      for (int n = 0; n < 2; n++) {
        int nTile = n * 4;
        load_matrix_x4(aReg, (As[mWarp + mTile + loadRowA] + loadColA));
        load_matrix_x2(bReg, (Bs[nWarp + nTile + loadRowB] + loadColB));
        mma_m16n8k16(aReg, bReg, dReg[m][n], dReg[m][n]);
        load_matrix_x4(aReg, (As[mWarp + mTile + loadRowA] + (loadColA^2)));
        load_matrix_x2(bReg, (Bs[nWarp + nTile + loadRowB] + (loadColB^2)));
        mma_m16n8k16(aReg, bReg, dReg[m][n], dReg[m][n]);
      }
    }
    __syncthreads();
  }

  int groupID     = laneID >> 2;
  int groupLaneID = laneID % 4;
  for (int m = 0; m < 2; m++) {
    for (int n = 0; n <  2; n++) {
      int mTile = m * 16;
      int nTile = n * 8;
      float2 d0 = make_float2(dReg[m][n][0], dReg[m][n][1]);
      float2 d2 = make_float2(dReg[m][n][2], dReg[m][n][3]);
      float2 *cOut0 = reinterpret_cast<float2 *>(C + (mBlock + mTile + 2*mWarp + groupID    )*N + nBlock + nTile + 2*nWarp + 2 * groupLaneID);
      float2 *cOut2 = reinterpret_cast<float2 *>(C + (mBlock + mTile + 2*mWarp + groupID + 8)*N + nBlock + nTile + 2*nWarp + 2 * groupLaneID);
      *cOut0 = d0;
      *cOut2 = d2;
    }
  }
  __syncthreads();
}

__launch_bounds__(16 * 16)
__global__ void matmul_v3(const half *A, const half *B, float *C, int M, int N, int K) {
  __shared__ uint4 As[32][8];
  __shared__ uint4 Bs[32][8];
  int mBlock = blockIdx.y*64;
  int nBlock = blockIdx.x*64;
  const uint4 *globalTileA = reinterpret_cast<const uint4 *>(A + mBlock * K);
  const uint4 *globalTileB = reinterpret_cast<const uint4 *>(B + nBlock * K);

  int threadID = threadIdx.y * blockDim.x + threadIdx.x;
  int warpID = threadID / 32;
  int laneID = threadID % 32;
  int mWarp = 16 * (warpID / 4);
  int nWarp = 8 * (warpID % 4);

  unsigned aReg[4];
  unsigned bReg[2];
  float dReg[2][2][4] = {0.};

  int storeRow = warpID * 4 + laneID / 8;
  // HINT: The code can be found in one of the readings/references ;)
  int storeCol = laneID % 8 ^ (laneID / 8);

  int loadRowA = (laneID % 16) / 2;
  int loadColA = (laneID / 16 + 4 * (laneID % 2)) ^ (loadRowA % 4);
  int loadRowB = (laneID % 8) / 2;
  int loadColB = (laneID / 8 + 4 * (laneID % 2)) ^ (loadRowB % 4);


  for (int k = 0; k < K/8; k += 4) {
      // async stuff
      cp_async(&As[storeRow][storeCol], &globalTileA[(warpID*8 + laneID/4)*K/8 + k + laneID%4]);
      cp_async(&Bs[storeRow][storeCol], &globalTileB[(warpID*8 + laneID/4)*K/8 + k + laneID%4]);
      asm volatile("cp.async.commit_group;\n" ::); //create new cp.async-group
      asm volatile("cp.async.wait_group 0;\n" ::);
      __syncthreads();

      for (int m = 0; m < 2; m++) {
          int mTile = m * 8; 
          for (int n = 0; n < 2; n++) {
              int nTile = n * 4;
              load_matrix_x4(aReg, (As[mWarp + mTile + loadRowA] + loadColA));
              load_matrix_x2(bReg, (Bs[nWarp + nTile + loadRowB] + loadColB));
              mma_m16n8k16(aReg, bReg, dReg[m][n], dReg[m][n]);
              load_matrix_x4(aReg, (As[mWarp + mTile + loadRowA] + (loadColA^2)));
              load_matrix_x2(bReg, (Bs[nWarp + nTile + loadRowB] + (loadColB^2)));
              mma_m16n8k16(aReg, bReg, dReg[m][n], dReg[m][n]);      
          }
      }
      __syncthreads();
  }

  int groupID     = laneID >> 2;
  int groupLaneID = laneID % 4;
  for (int m = 0; m < 2; m++) {
      for (int n = 0; n < 2; n++) {
          int mTile = m * 16;
          int nTile = n * 8;
          float2 d0 = make_float2(dReg[m][n][0], dReg[m][n][1]);
          float2 d2 = make_float2(dReg[m][n][2], dReg[m][n][3]);
          float2 *cOut0 = reinterpret_cast<float2 *>(C + (mBlock + mTile + 2*mWarp + groupID    )*N + nBlock + nTile + 2*nWarp + 2 * groupLaneID);
          float2 *cOut2 = reinterpret_cast<float2 *>(C + (mBlock + mTile + 2*mWarp + groupID + 8)*N + nBlock + nTile + 2*nWarp + 2 * groupLaneID);
          *cOut0 = d0;
          *cOut2 = d2;
      }
  }
  __syncthreads();
}


torch::Tensor launch_matmul_v1(torch::Tensor A, torch::Tensor B, torch::Tensor C, const int M, const int N, const int K) {
    const int mma_m_tiles = 2;
    const int mma_n_tiles = 2;
    const int BK_SIZE = 16;
    const int NUM_WARPS_M = 4;
    const int NUM_WARPS_N = 2;

    dim3 dim_block(16,16);
    dim3 dim_grid((M/64),(N/64));

    matmul_v1<<<dim_grid, dim_block>>>(
            reinterpret_cast<half*>(A.data_ptr<at::Half>()),
            reinterpret_cast<half*>(B.data_ptr<at::Half>()),
            C.data_ptr<float>(),
            M, N, K
            );

    return C;
}

torch::Tensor launch_matmul_v2(torch::Tensor A, torch::Tensor B, torch::Tensor C, const int M, const int N, const int K) {
    const int mma_m_tiles = 2;
    const int mma_n_tiles = 2;
    const int BK_SIZE = 16;
    const int NUM_WARPS_M = 4;
    const int NUM_WARPS_N = 2;

    dim3 dim_block(16,16);
    dim3 dim_grid((M/64),(N/64));

    matmul_v2<<<dim_grid, dim_block>>>(
            reinterpret_cast<half*>(A.data_ptr<at::Half>()),
            reinterpret_cast<half*>(B.data_ptr<at::Half>()),
            C.data_ptr<float>(),
            M, N, K
            );

    return C;
}


torch::Tensor launch_matmul_v3(torch::Tensor A, torch::Tensor B, torch::Tensor C, const int M, const int N, const int K) {

    dim3 dim_block(16,16);
    dim3 dim_grid((M/64),(N/64));

    matmul_v3<<<dim_grid, dim_block>>>(
            reinterpret_cast<half*>(A.data_ptr<at::Half>()),
            reinterpret_cast<half*>(B.data_ptr<at::Half>()),
            C.data_ptr<float>(),
            M, N, K
            );

    return C;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_v1", &launch_matmul_v1, "HGEMM Naive");
    m.def("matmul_v2", &launch_matmul_v2, "HGEMM Permuted Shared");
    m.def("matmul_v3", &launch_matmul_v3, "HGEMM Async");
}
