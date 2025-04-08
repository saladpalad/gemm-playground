`usage: python3 test_sgemm.py [-h] (-v1 | -v2 | -v3 | -v4 | -all)`

Output on UCLA AI Safety Server 3090 GPU:

```
Compiling SGEMM module...                                      
                                                               
------------                                                   
STUDENT Matmul_v1 (NAIVE) TEST CHECK: True                     
STUDENT Matmul_v1 time: 254.5172 ms                            
STUDENT Matmul_v1 GLFOPS/s: 539.9986                           
                                                               
STUDENT Matmul_v2 (COALESCED) TEST CHECK: True                 
STUDENT Matmul_v2 time: 63.0671 ms                             
STUDENT Matmul_v2 GLFOPS/s: 2179.2484                          
                                                               
STUDENT Matmul_v3 (REGISTER REUSE) TEST CHECK: True            
STUDENT Matmul_v3 time: 12.2030 ms                             
STUDENT Matmul_v3 GLFOPS/s: 11262.7114                         
                                                               
STUDENT Matmul_v4 (SHARED MEMORY) TEST CHECK: True             
STUDENT Matmul_v4 time: 10.2973 ms                             
STUDENT Matmul_v4 GLFOPS/s: 13347.0292
```
