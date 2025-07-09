import numpy as np
import matplotlib.pyplot as plt

from utils.copula_utils import (
    Clayton_copula_pdf_from_PITs,
    sJoe_copula_pdf_from_PITs,
    sGumbel_copula_pdf_from_PITs,
)
from utils.scoring import LogS

def clipped_plot_copula(pdf_func, theta, title, grid_size=200):
    u_seq = np.linspace(0.005, 0.995, grid_size)
    U1, U2 = np.meshgrid(u_seq, u_seq)
    grid = np.column_stack([U1.ravel(), U2.ravel()])
    pdf_vals = np.clip(pdf_func(grid, theta).reshape(U1.shape), 0, 10)

    plt.figure(figsize=(6, 5))
    plt.contourf(U1, U2, pdf_vals, levels=30, cmap="viridis")
    plt.title(f"{title} (theta={theta})")
    plt.xlabel("u1")
    plt.ylabel("u2")
    plt.colorbar(label="density")
    plt.tight_layout()
    plt.show()

clipped_plot_copula(Clayton_copula_pdf_from_PITs, theta=3,   title="Clayton")
clipped_plot_copula(sJoe_copula_pdf_from_PITs,   theta=2.5, title="Survival Joe")
clipped_plot_copula(sGumbel_copula_pdf_from_PITs ,theta=2,   title="Survival Gumbel")


u_seq = np.linspace(0.005, 0.995, 200)
U1, U2 = np.meshgrid(u_seq, u_seq)
grid = np.column_stack([U1.ravel(), U2.ravel()])
score_diff = LogS(Clayton_copula_pdf_from_PITs(grid, theta=3)) - LogS(sJoe_copula_pdf_from_PITs(grid, theta=2.5))
print(score_diff)
print(score_diff.min(), score_diff.max())
print(score_diff.mean())
plt.figure(figsize=(6, 5))
plt.contourf(U1, U2, score_diff.reshape(U1.shape), levels=30, cmap="viridis")
plt.title(f"score differences Clayton - sJoe")
plt.xlabel("u1")
plt.ylabel("u2")
plt.colorbar(label="log score difference")
plt.tight_layout()
plt.show()