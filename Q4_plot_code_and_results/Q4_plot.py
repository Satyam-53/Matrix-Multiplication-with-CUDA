
import re
import matplotlib
matplotlib.use("Agg")  # Save plots without a display (for EC2)
import matplotlib.pyplot as plt

# Read CUDA results from file
with open("Q4_results.txt", "r") as f:
    text = f.read()

matches = re.findall(
    r"N:\s*(\d+),\s*GFLOPS:\s*([\d.]+)", text
)

cuda = {int(n): float(gf) for n, gf in matches}

# Given benchmark values
blas = {
    255: 5.93, 256: 5.84, 512: 17.4, 768: 45.3,
    1023: 73.7, 1024: 73.6, 1025: 73.5,
    2047: 171, 2048: 182, 2049: 175,
    4095: 209.2, 4096: 213.4
}

cublas = {
    255: 2456.2, 256: 2515.3, 512: 4573.6,
    768: 4333.1, 1023: 4222.5, 1024: 4404.9,
    1025: 3551.0, 2047: 4490.5, 2048: 4669.8,
    2049: 4120.7, 4095: 4389.0, 4096: 4501.6
}

# ==========================================
# Plot 1: CUDA vs BLAS vs cuBLAS
# Only values present in the assignment table
# ==========================================

common_n = sorted(set(cuda) & set(blas) & set(cublas))
x = range(len(common_n))

plt.figure(figsize=(12, 6))

# plt.plot(
#     x, [cuda[n] for n in common_n],
#     "o-", label="Our CUDA Kernel", linewidth=2
# )

plt.plot(
    x, [blas[n] for n in common_n],
    "s--", label="Multi-core BLAS", linewidth=2, color='orange'
)

# plt.plot(
#     x, [cublas[n] for n in common_n],
#     "^--", label="cuBLAS", linewidth=2
# )

plt.xlabel("Matrix Size (N)")
plt.ylabel("Performance (GFLOPS)")
plt.title("CUDA vs BLAS vs cuBLAS (Block Size: 32 × 8)")
plt.xticks(x, common_n, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("Q4_BLAS.png", dpi=300)
plt.close()


# ==========================================
# Plot 2: CUDA only
# All values of N between 255 and 2049
# ==========================================

# all_n = sorted(cuda)
# x = range(len(all_n))

# plt.figure(figsize=(12, 6))

# plt.plot(
#     x, [cuda[n] for n in all_n],
#     "o-", label="Our CUDA Kernel", linewidth=2
# )

# plt.xlabel("Matrix Size (N)")
# plt.ylabel("Performance (GFLOPS)")
# plt.title("CUDA Performance vs Matrix Size (Block Size: 32 × 8)")
# plt.xticks(x, all_n, rotation=45)
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()

# plt.savefig("Q4_cuda_performance.png", dpi=300)
# plt.close()

# print("Saved Q4_comparison.png")
# print("Saved Q4_cuda_performance.png")
