import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm
from statsmodels.distributions.empirical_distribution import ECDF

# Simulate marginals
n = 10000
df = 5
y1 = t.rvs(df, loc=0, scale=1, size=n)
y2 = t.rvs(df, loc=0, scale=1, size=n)

# Calc PITs
u1 = t.cdf(y1, df)
u2 = t.cdf(y2, df)

# Use quantile to calc ECDF-wise inversion
F1_inv = lambda u: np.quantile(y1, u)
F2_inv = lambda u: np.quantile(y2, u)

# Create restriction and its 5% quantile threshold
restr = y1 + y2
q_05 = np.quantile(restr, 0.05)

# Create a grid over [0,1]^2 and create weights
grid_size = 100
u_vals = np.linspace(0.001, 0.999, grid_size)   # instead of 0 and 1 to avoid edge cases messing up calculations for now
U1, U2 = np.meshgrid(u_vals, u_vals)
W = np.zeros_like(U1)

for i in range(grid_size):
    for j in range(grid_size):
        yi = F1_inv(U1[i, j])
        yj = F2_inv(U2[i, j])
        W[i, j] = 1.0 if (yi + yj) <= q_05 else 0.0

# Plot the weight region in the copula space
plt.figure(figsize=(6, 5))
plt.contourf(U1, U2, W, levels=[0, 0.5, 1])
plt.title("Localized Region in Copula Domain for y1 + y2 ≤ q_0.05")
plt.xlabel("u1 (PIT of y1)")
plt.ylabel("u2 (PIT of y2)")
plt.grid(True)
plt.colorbar(label="Weight (1 = in region)")
plt.show()