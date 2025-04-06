# gemm-playground

Some testing I did for various GEMM Kernels \
**SGEMM** has input matrices FP32 \
**HGEMM** utilizes 3rd gen tensor cores w/ input matrices being FP16 \
**DGEMM** is a distributed gemm implementation that implements some GEMM kernel w/ communication primitives from NCCL

#### Some TODOs for me
- HGEMM w/ 4th gen tensor cores (Hopper + TMA)
- Make DGEMM better lol

