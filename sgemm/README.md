`usage: python3 test_sgemm.py [-h] (-v1 | -v2 | -v3 | -v4 | -all)`

Output on UCLA AI Safety Server 3090 GPU:

```
Compiling SGEMM module...

------------
STUDENT Matmul_v1 (NAIVE) TEST CHECK: True
STUDENT Matmul_v1 time: 253.9735 ms
STUDENT Matmul_v1 GLFOPS/s: 541.1548

STUDENT Matmul_v2 (COALESCED) TEST CHECK: False
STUDENT Matmul_v2 time: 13.7370 ms
STUDENT Matmul_v2 GLFOPS/s: 10005.0484

STUDENT Matmul_v3 (REGISTER REUSE) TEST CHECK: True
STUDENT Matmul_v3 time: 11.4954 ms
STUDENT Matmul_v3 GLFOPS/s: 11955.9705

STUDENT Matmul_v4 (SHARED MEMORY) TEST CHECK: True
STUDENT Matmul_v4 time: 9.8967 ms
STUDENT Matmul_v4 GLFOPS/s: 13887.3010
```
