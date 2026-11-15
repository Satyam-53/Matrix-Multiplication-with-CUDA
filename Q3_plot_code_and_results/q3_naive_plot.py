import matplotlib.pyplot as plt

N = [256, 512, 1024, 2048]
gflops = [318, 350, 310, 195]
block_sizes = ["32 × 8", "32 × 8", "32 × 8", "16 × 16"]

# Equally spaced x-axis positions
x = range(len(N))

plt.figure(figsize=(8, 5))

plt.plot(x, gflops, marker='o', linewidth=1.8)

# Annotate each point with its block size
for xi, yi, label in zip(x, gflops, block_sizes):
    plt.annotate(
        label,
        (xi, yi),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        fontsize=10
    )

plt.xticks(x, N)
plt.xlabel("Matrix size (N)")
plt.ylabel("Performance (GFLOP/s)")
plt.title("Naive matrix multiplication")

plt.ylim(0, 440)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig('Naive.png')