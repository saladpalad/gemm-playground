import torch
import os
from torch.utils.cpp_extension import load

# Set environment variables
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'  # Ampere
print("\nCompiling DGEMM module...\n")
dgemm = load(name="dgemm", 
             sources=["dgemm.cu"], 
             extra_cuda_cflags=["-arch=sm_86", "-O3"])

def time_matmul(func, *args, num_runs=1):
    # First do a warmup run without timing
    _ = func(*args)
    times = []
    for _ in range(num_runs):
        # Synchronize all devices before starting
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
            
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        result = func(*args)
        end_event.record()
        
        # Synchronize all devices after completion
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
            
        times.append(start_event.elapsed_time(end_event))
    
    avg_time = sum(times) / len(times)
    return result, avg_time

if __name__ == "__main__":
    # Test parameters for v1 and v2
    m, n, k = 16384, 16384, 16384
    
    # Generate random matrices for v1 and v2
    A_large = torch.rand(m, k, device='cuda')
    B_large = torch.rand(k, n, device='cuda')
    C_v1 = torch.zeros(m, n, device='cuda')
    C_v2 = torch.zeros(m, n, device='cuda')
    
    
    # Calculate theoretical GFLOPS for large matrices
    NUM_GFLOPS = 2 * m * n * k / 1e9
    
    # Run distributed matmul with large matrices for v1 and v2
    print("\nRunning v1 and v2 with large 8384x8384 matrices...")
    C_v1, time_v1 = time_matmul(dgemm.distributed_matmul_v1, A_large, B_large, C_v1, m, n, k)
    C_v2, time_v2 = time_matmul(dgemm.distributed_matmul_v2, A_large, B_large, C_v2, m, n, k)
    
    
    # Print results
    print('\nDistributed GEMM V1 TEST CHECK:', torch.allclose(A_large @ B_large, C_v1, rtol=1e-03, atol=1e-03))
    print(f"Distributed GEMM V1 time: {time_v1:.4f} ms")
    print(f"Distributed GEMM V1 GFLOPS/s: {(NUM_GFLOPS/(time_v1*1e-3)):.4f}\n")
    
    print('Distributed GEMM V2 TEST CHECK:', torch.allclose(A_large @ B_large, C_v2, rtol=1e-03, atol=1e-03))
    print(f"Distributed GEMM V2 time: {time_v2:.4f} ms")
    print(f"Distributed GEMM V2 GFLOPS/s: {(NUM_GFLOPS/(time_v2*1e-3)):.4f}\n")
