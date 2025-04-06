import torch
from torch.utils.cpp_extension import load
import os
#import hgemm_ref as mm

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'  # Ampere

print("\nCompiling tensor core matmul module...\n")
ms = load(name="hgemm", sources=["hgemm.cu"], extra_cuda_cflags=["-arch=sm_86", "-O3", "-g", "--generate-line-info"])

def time_matmul(func, *args, num_runs=1):
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_runs):
        result = func(*args)
    end_event.record()

    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / num_runs
    return result, elapsed_time

def matmul_ref(A, B, C):
    C = A@B
    return C

if __name__ == "__main__":
    m, n, k = 4096, 4096, 4096 
    A, B = torch.rand(m,k, device='cuda'), torch.rand(k,n, device='cuda')
    A_half, B_half = A.half(), B.half()
    BT = torch.transpose(B,0,1).contiguous()
    BT_half = BT.half()

    NUM_GFLOPS = 2*m*n*k/1e9

    C_v1 = torch.zeros(m,n, device='cuda')
    C_v2 = torch.zeros(m,n, device='cuda')
    C_v3 = torch.zeros(m,n, device='cuda')
#    C_ref = torch.zeros(m,n, device='cuda')

    C_v1, C_v1_time = time_matmul(ms.matmul_v1, A_half, BT_half, C_v1, m, n, k)
    C_v2, C_v2_time = time_matmul(ms.matmul_v2, A_half, BT_half, C_v2, m, n, k)
    C_v3, C_v3_time = time_matmul(ms.matmul_v3, A_half, BT_half, C_v3, m, n, k)
#    C_ref, C_ref_time = time_matmul(matmul_ref, A_half, BT_half, C_ref)

    print('Matmul_v1 (MMA NAIVE) TEST CHECK:', torch.allclose(A@B, C_v1, rtol=1e-03, atol=1e-08))
    print(f"Matmul_v1 time: {C_v1_time:.4f} ms")
    print(f"Matmul_v1 GLFOPS/s: {(NUM_GFLOPS/(C_v1_time*1e-3)):.4f}\n")

    print('Matmul_v2 (MMA PERMUTED SHARED) TEST CHECK:', torch.allclose(A@B, C_v2, rtol=1e-03, atol=1e-08))
    print(f"Matmul_v2 time: {C_v2_time:.4f} ms")
    print(f"Matmul_v2 GLFOPS/s: {(NUM_GFLOPS/(C_v2_time*1e-3)):.4f}\n")

#    print(f"Matmul_ref time: {C_ref_time:.4f} ms")
#    print(f"Matmul_ref GLFOPS/s: {(NUM_GFLOPS/(C_ref_time*1e-3)):.4f}\n")


    print('Matmul_v3 (MMA CP ASYNC) TEST CHECK:', torch.allclose(A@B, C_v3, rtol=1e-03, atol=1e-08))
    print(f"Matmul_v3 time: {C_v3_time:.4f} ms")
    print(f"Matmul_v3 GLFOPS/s: {(NUM_GFLOPS/(C_v3_time*1e-3)):.4f}\n")
