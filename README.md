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

4. Not much gain due to k unrolling with TILED_DIM_K, even with explicit register allocations, and separating load and compute phases, for prefetching loads. Probably compiler was already optimizing it.

5. Profiling:
- 2 blocks allowed but register and shared memory bottleneck
    - 72 threads/reg
    - 64 KB per block

- Bank conflicts found during store phase

==PROF== Disconnected from process 3164
[3164] mmpy@127.0.0.1
  matMul(int, float *, float *, float *) (16, 16, 1)x(32, 16, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                   119473
    l1tex__t_requests_pipe_lsu_mem_shared_op_ld.sum                           (!) n/a
    l1tex__t_requests_pipe_lsu_mem_shared_op_st.sum                           (!) n/a
    -------------------------------------------------------- ----------- ------------


- Reduced dynamic smem to 32
- Changed k loop to do load and compute in same loop: threads/reg went from 72 to 68
- BLOCKDIM_X and BLOCKDIM_Y: 64 reg/thread - occupancy went from 49.96% to 93.10%

Occupany increased but performance decreased: N: 256, GFLOPS: 567.718791
N: 512, GFLOPS: 1849.983416
N: 1024, GFLOPS: 2267.901917
N: 1025, GFLOPS: 2003.151720
N: 2047, GFLOPS: 2421.096829
N: 2048, GFLOPS: 2184.231963

Current bottleneck: MIO throttling happens when a GPU warp wants to execute a memory-related instruction, but the MIO instruction queue is full, so it must wait.

Increase FMA per load/store: decerease by to 8



6. #define TILEDIM_K 32 //better perf reason?
sudo "$(which ncu)"   --section LaunchStats   --section Occupancy   --launch-count 1   ./mmpy -n 1024
==PROF== Connected to process 11841 (/home/ubuntu/cse260-parallel-computation-hw2-group-14/mmpy)
# devices: 1

Device is a Tesla T4, capability: 7.5
Clock speed: 1590.000000 MHz
# cores: 40

Global memory: 14.563171GB
Memory Clock Rate (MHz): 5001.000000
Memory Bus Width (bits): 256
Peak Memory Bandwidth (GB/s): 320.064000
L2 Cache size: (KB): 4096.000000
ECC Enabled
Device shares a unified address space with the host
CUDA Driver version: 13020, runtime version: 13040

 capability 7.5
n: 1024, tx: 32, ty: 8, gridX: 16, gridY: 16, reps: 10, epsilon: 0.000366211

Using Single precision arithmetic


==PROF== Profiling "matMul" - 0 (1/1): 0%....50%....100% - 1 pass
Device computation time: 0.663117 sec. [32.384690 gflops]

     N     TX     TY   Reps       t_h     GF_h       t_d   GF_d   Host?  L1? Rnd?  BT?  SHM
@  1024    32      8     10   0.00e+00     0.0  6.63e-01   32.4     N    N    N    N    4


Device result: answers matched to within 0.000366211

==PROF== Disconnected from process 11841
[11841] mmpy@127.0.0.1
  matMul(int, float *, float *, float *) (16, 16, 1)x(32, 8, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: Launch Statistics
    -------------------------------- --------------- -----------------
    Metric Name                          Metric Unit      Metric Value
    -------------------------------- --------------- -----------------
    Block Size                                                     256
    Function Cache Configuration                     CachePreferShared
    Grid Size                                                      256
    Registers Per Thread             register/thread                74
    Shared Memory Configuration Size           Kbyte             65.54
    Driver Shared Memory Per Block        byte/block                 0
    Dynamic Shared Memory Per Block      Kbyte/block             16.38
    Static Shared Memory Per Block        byte/block                 0
    # SMs                                         SM                40
    Stack Size                                                    1024
    Threads                                   thread             65536
    # TPCs                                                          20
    Enabled TPC IDs                                                all
    Uses Green Context                                               0
    Waves Per SM                                                  2.13
    -------------------------------- --------------- -----------------

    OPT   Est. Speedup: 33.33%                                                                                          
          A wave of thread blocks is defined as the maximum number of blocks that can be executed in parallel on the    
          target GPU. The number of blocks in a wave depends on the number of multiprocessors and the theoretical       
          occupancy of the kernel. This kernel launch results in 2 full waves and a partial wave of 16 thread blocks.   
          Under the assumption of a uniform execution duration of all thread blocks, this partial wave may account for  
          up to 33.3% of the total runtime of this kernel. Try launching a grid with no partial wave. The overall       
          impact of this tail effect also lessens with the number of full waves executed for a grid. See the Hardware   
          Model (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model) description for     
          more details on launch configurations.                                                                        

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block            3
    Block Limit Shared Mem                block            4
    Block Limit Warps                     block            4
    Theoretical Active Warps per SM        warp           24
    Theoretical Occupancy                     %           75
    Achieved Occupancy                        %        66.04
    Achieved Active Warps Per SM           warp        21.13
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 25%                                                                                       
          The 6.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the       
          hardware maximum of 8. This kernel's theoretical occupancy (75.0%) is limited by the number of required       
          registers.                                                                                                    

ubuntu@ip-172-31-37-150:~/cse260-parallel-computation-hw2-group-14$ ./grade.sh 
N: 256, GFLOPS: 1148.877456
N: 512, GFLOPS: 2907.799346
N: 1024, GFLOPS: 3723.213978
N: 1025, GFLOPS: 3334.051785
N: 2047, GFLOPS: 3469.593954
N: 2048, GFLOPS: 3764.056606
ubuntu@ip-172-31-37-150:~/cse260-parallel-computation-hw2-group-14$ 