# Matrix Multiplication with CUDA
## Overview
This CSE260 project optimizes an FP32 CUDA matrix multiplication kernel on an NVIDIA Tesla T4, progressing from a naive implementation to a tiled kernel with multiple output elements computed per thread.

The optimizations include:
- Shared-memory tiling: Cooperatively load and reuse tiles of both input matrices to reduce global-memory traffic.
- Register blocking: Accumulate multiple output elements per thread to increase data reuse and instruction-level parallelism.
- Thread-block and tile tuning: Compare configurations to balance register usage, shared-memory consumption, and occupancy.
- Profiling-guided refinement: Use Nsight Compute to investigate shared-memory bank conflicts, register pressure, and MIO throttling.
The optimized kernel achieved 3,764 GFLOP/s (3.76 TFLOP/s) for 2048 × 2048 matrices in a recorded benchmark run. Experiments also showed that increasing occupancy alone did not necessarily improve performance.

The repository includes the kernel implementation, benchmark plots, comparisons with cuBLAS, and [profiling notes](findings.txt).

## Commands to build and run

To build the binary:
```bash
make `cat src_todo_T4/OPTIONS.txt`
```

To build the binary with cublas:
```bash
make cublastest=1 `cat src_todo_T4/OPTIONS.txt`
```

To clean the output files of the make command:
```bash
make clean
```

> [!IMPORTANT]
> On the AWS g4dn.xlarge instance run the following command before running any experiments. It sets fixed GPU clock rates to ensure consistent performance
> ```bash
> sudo nvidia-smi -pm 1
> sudo nvidia-smi -lgc 1590
> ```
To Run:
```bash
./mmpy `cat src_todo_T4/OPTIONS_RUNTIME.txt` -n 256
```

## Additional Useful Commands 
To run Script in tools folder:
```bash
./tools/run_ncu.sh
```

If you get Permission denied error when executing a file:
```bash
chmod +x name_of_file
```
eg: `chmod +x tools/*`

Find GPU chipset:
```bash
lspci | grep -i --color 'vga\|3d\|2d'
```





