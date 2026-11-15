import matplotlib.pyplot as plt

N = [256, 512, 1024, 1025, 2047, 2048]
x = range(len(N))

perf_16_16 = [1068.62, 2756.18, 3572.02, 3178.05, 3631.69, 3989.57]
perf_32_16 = [1165.05, 2180.71, 2241.15, 2115.90, 2303.98, 2350.02]
perf_32_8  = [1165.05, 2889.89, 3727.68, 3299.35, 3503.47, 3755.07]

plt.figure(figsize=(10, 6))

plt.plot(x, perf_16_16, marker='o', linewidth=2, label='bx=16, by=16')
plt.plot(x, perf_32_16, marker='s', linewidth=2, label='bx=32, by=16')
plt.plot(x, perf_32_8, marker='^', linewidth=2, label='bx=32, by=8')

plt.xticks(x, N)
plt.xlabel('Matrix Size (N)')
plt.ylabel('Performance (GFLOP/s)')
plt.title('CUDA Matrix Multiplication Performance')
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig('Q2_performance_comparison.png')