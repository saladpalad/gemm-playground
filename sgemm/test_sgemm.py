import torch
import os
from torch.utils.cpp_extension import load
#import sgemm_ref as mm
import argparse


# ncu --set full --import-source yes --export reports/sgemm_all_2 python3 test_sgemm.py -all

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'  # Ampere

print("\nCompiling SGEMM module...\n")
ms = load(name="matmul_module", sources=["sgemm.cu"], extra_cuda_cflags=["-O3", "--generate-line-info"])

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test SGEMM implementations')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-v1', action='store_true', help='Run naive implementation')
    group.add_argument('-v2', action='store_true', help='Run coalesced implementation')
    group.add_argument('-v3', action='store_true', help='Run register reuse implementation')
    group.add_argument('-v4', action='store_true', help='Run shared memory implementation')
#    group.add_argument('-cublas', action='store_true', help='Run CUBLAS SGEMM')
    group.add_argument('-all', action='store_true', help='Run all implementations')
    
    args = parser.parse_args()
    
    m, n, k = 4096, 4096, 4096 
    
    A, B = torch.rand(m,k, device='cuda').contiguous(), torch.rand(k,n, device='cuda').contiguous()
    AT = torch.transpose(A,0,1).contiguous()
    BT = torch.transpose(B,0,1).contiguous()
    
    NUM_GFLOPS = 2*m*n*k/1e9
    C_cublas_time = 16.5979
    CUBLAS_GFLOPS = NUM_GFLOPS/(C_cublas_time*1e-3)

    if args.v1 or args.all:
        C_v1 = torch.zeros(m,n, device='cuda')
        C_v1_ref = torch.zeros(m,n, device='cuda')
        C_v1, C_v1_time = time_matmul(ms.matmul_v1, A, B, C_v1, m, n, k) #MODIFY THIS LINE TO USE A, B, AT or BT
#        C_v1_ref, C_v1_time_ref = time_matmul(mm.matmul_v1, A, B, C_v1, m, n, k)
        STUDENT_GFLOPS = NUM_GFLOPS/(C_v1_time*1e-3)
        print('------------')
        print('STUDENT Matmul_v1 (NAIVE) TEST CHECK:', torch.allclose(A@B, C_v1, rtol=1e-05, atol=1e-08))
        print(f"STUDENT Matmul_v1 time: {C_v1_time:.4f} ms")
        print(f"STUDENT Matmul_v1 GLFOPS/s: {(NUM_GFLOPS/(C_v1_time*1e-3)):.4f}\n")

#        print(f"REFERENCE Matmul_v1 (NAIVE) time: {C_v1_time_ref:.4f} ms")
#        print(f"REFERENCE Matmul_v1 GLFOPS/s: {(NUM_GFLOPS/(C_v1_time_ref*1e-3)):.4f}")
#        print('------------')

    if args.v2 or args.all:
        C_v2 = torch.zeros(m,n, device='cuda')
        C_v2_ref = torch.zeros(m,n, device='cuda')
        C_v2, C_v2_time = time_matmul(ms.matmul_v2, A, B, C_v2, m, n, k) #MODIFY THIS LINE TO USE A, B, AT or BT
#        C_v2_ref, C_v2_time_ref = time_matmul(mm.matmul_v2, A, B, C_v2, m, n, k)
        STUDENT_GFLOPS = NUM_GFLOPS/(C_v2_time*1e-3)
        if not args.all:
            print('------------')
        print('STUDENT Matmul_v2 (COALESCED) TEST CHECK:', torch.allclose(A@B, C_v2, rtol=1e-05, atol=1e-08))
        print(f"STUDENT Matmul_v2 time: {C_v2_time:.4f} ms")
        print(f"STUDENT Matmul_v2 GLFOPS/s: {(NUM_GFLOPS/(C_v2_time*1e-3)):.4f}\n")
        
#        print(f"REFERENCE Matmul_v2 (COALESCED) time: {C_v2_time_ref:.4f} ms")
#        print(f"REFERENCE Matmul_v2 GLFOPS/s: {(NUM_GFLOPS/(C_v2_time_ref*1e-3)):.4f}")
#        print('------------')
    
    if args.v3 or args.all:
        C_v3 = torch.zeros(m,n, device='cuda')
        C_v3_ref = torch.zeros(m,n, device='cuda')
        C_v3, C_v3_time = time_matmul(ms.matmul_v3, A, B, C_v3, m, n, k) #MODIFY THIS LINE TO USE A, B, AT or BT
#        C_v3_ref, C_v3_time_ref = time_matmul(mm.matmul_v3, AT, B, C_v3, m, n, k)
        STUDENT_GFLOPS = NUM_GFLOPS/(C_v3_time*1e-3)
        if not args.all:
            print('------------')
        print('STUDENT Matmul_v3 (REGISTER REUSE) TEST CHECK:', torch.allclose(A@B, C_v3, rtol=1e-05, atol=1e-08))
        print(f"STUDENT Matmul_v3 time: {C_v3_time:.4f} ms")
        print(f"STUDENT Matmul_v3 GLFOPS/s: {(NUM_GFLOPS/(C_v3_time*1e-3)):.4f}\n")

#        print(f"REFERENCE Matmul_v3 (REGISTER REUSE) time: {C_v3_time_ref:.4f} ms")
#        print(f"REFERENCE Matmul_v3 GLFOPS/s: {(NUM_GFLOPS/(C_v3_time_ref*1e-3)):.4f}")
#        print('------------')
    
    if args.v4 or args.all:
        C_v4 = torch.zeros(m,n, device='cuda')
        C_v4_ref = torch.zeros(m,n, device='cuda')
        C_v4, C_v4_time = time_matmul(ms.matmul_v4, AT, B, C_v4, m, n, k) #MODIFY THIS LINE TO USE A, B, AT or BT
#        C_v4_ref, C_v4_time_ref = time_matmul(mm.matmul_v4, AT, B, C_v4, m, n, k)
        STUDENT_GFLOPS = NUM_GFLOPS/(C_v4_time*1e-3)
        if not args.all:
            print('------------')
        print('STUDENT Matmul_v4 (SHARED MEMORY) TEST CHECK:', torch.allclose(A@B, C_v4, rtol=1e-05, atol=1e-08))
        print(f"STUDENT Matmul_v4 time: {C_v4_time:.4f} ms")
        print(f"STUDENT Matmul_v4 GLFOPS/s: {(NUM_GFLOPS/(C_v4_time*1e-3)):.4f}\n")
    
#        print(f"REFERENCE Matmul_v4 (SHARED MEMORY) time: {C_v4_time_ref:.4f} ms")
#        print(f"REFERENCE Matmul_v4 GLFOPS/s: {(NUM_GFLOPS/(C_v4_time_ref*1e-3)):.4f}")
#        print('------------')


