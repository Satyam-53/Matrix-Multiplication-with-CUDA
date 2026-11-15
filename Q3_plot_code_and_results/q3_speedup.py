import matplotlib.pyplot as plt
import numpy as np

N = [256, 512, 1024, 2048]

optimized = np.array([1165.05, 2889.89, 3727.68, 3989.57])
naive = np.array([318, 350, 310, 195])

block_sizes = ["32×8", "32×8", "32×8", "16×16"]

speedup = optimized / naive
x = np.arange(len(N))

plt.figure(figsize=(8, 5))

bars = plt.bar(x, speedup, width=0.6, color="steelblue")

for bar, s, block in zip(bars, speedup, block_sizes):
    plt.annotate(
        f"{s:.2f}×\n({block})",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.xticks(x, N)
plt.xlabel("Matrix Size (N)")
plt.ylabel("Speedup over Naive Kernel (×)")
plt.title("Speedup of Optimized CUDA Matrix Multiplication")

plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.gca().set_axisbelow(True)
plt.ylim(0, max(speedup) * 1.2)

plt.tight_layout()
plt.savefig("speedup.png", dpi=300, bbox_inches="tight")
