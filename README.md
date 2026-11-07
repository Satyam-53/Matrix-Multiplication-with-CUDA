# Matrix Multiplication with CUDA
Starter code for CSE260 HW2.

You will only need to edit files in src_todo folder.
Run all your commands from the root directory of the git repo.

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





<!--  -->
1. when and how bank conflict was occuring with accessing the A, with what block size and tile sizes.
    bx=16, by=16 in starter code
    In our case, we saw that while computing C, we see every 2 rows of A with same column, were being accessed by threads of same warp
    ---> bank conflict for A. Min value for bx for no bankk conflicts is 32. If that is maintained, then one warp threads only access one row of A. 
    And since col size is so large, and every col elemnt in one row is placed on a different bank, there are no bank conflicts.

2. TLP vs ILP tradeoffs
    We started off with bx=32, by=32, TILED_DIMS: M=N=32, K=128
    Interleaving:
    bx=32, by=32, TILED_DIMS: M=N=64, K=128, TILESCALES: M=N=2
    Increased perf - threads same but interleaving improves performance
    Now bx has to stay at 32, so that A access does not create bank conflict for a warp
    But to increase ILP, we can reduce by
    We tried by = 4,8,16 and found best results when by=16. by=8 gave similar results, and by=4 gave bad results due to too less threads.

3. thought process: 
- We checked that loading of As and Bs is also bank conflict free
- What if we now load A in col major order since outer pdt requires that?
    2 methods:
    1) We take col elemnts of global A and put it in contiguous places in As : disrupts global memory colleascing
    2) We take row elemnts of gloabl A and put it in columns of As, but since column length of As is a multiple/divisor of 32, when storing col wise in As, we will get bank conflicts
    So we leave A in the row major order