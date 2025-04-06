`usage: python3 test_hgemm.py`

Output on UCLA AI Safety 3090 GPU (the PTX instructions used are for 3rd gen tensor core programming)
```
Compiling tensor core matmul module...

Matmul_v1 (MMA NAIVE) TEST CHECK: True
Matmul_v1 time: 6.7031 ms
Matmul_v1 GLFOPS/s: 20503.7775

Matmul_v2 (MMA PERMUTED SHARED) TEST CHECK: True
Matmul_v2 time: 3.9762 ms
Matmul_v2 GLFOPS/s: 34565.4721

Matmul_v3 (MMA CP ASYNC) TEST CHECK: True
Matmul_v3 time: 2.9870 ms
Matmul_v3 GLFOPS/s: 46012.2467
```
