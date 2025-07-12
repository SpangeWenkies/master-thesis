import matplotlib.pyplot as plt
import numpy as np

from utils.divergences import full_kl
from utils.copula_utils import sim_sGumbel_PITs, sGumbel_copula_pdf_from_PITs, sJoe_copula_pdf_from_PITs, Clayton_copula_pdf_from_PITs

# Example placeholder data again after code reset
theta_J = np.linspace(1.5, 15, 100)
theta_sGumbel = 1.5
pdf_sGumbel = lambda u: sGumbel_copula_pdf_from_PITs(u, theta_sGumbel)
pdf_sJoe = np.empty(100, dtype=object)
for i in range(100):
    theta = theta_J[i]
    pdf_sJoe[i] = lambda u, theta=theta: sJoe_copula_pdf_from_PITs(u, theta)
u_big = sim_sGumbel_PITs(1000, theta_sGumbel)
kl_values = np.array([full_kl(u_big, pdf_sGumbel, pdf_sJoe[i]) for i in range(len(theta_J))])

# Plotting KL vs theta_J
plt.figure(figsize=(6, 4))
plt.plot(theta_J, kl_values, marker='o', linestyle='-')
plt.xlabel(r"theta_{sJoe}")
plt.ylabel("KL divergence to true (sGumbel)")
plt.title(r"Monotonicity check: theta_{sJoe} vs KL distance")
plt.grid(True)
plt.tight_layout()
plt.show()

pdf_Clayton = np.empty(100, dtype=object)
for i in range(100):
    theta = theta_J[i]
    pdf_Clayton[i] = lambda u, theta=theta: Clayton_copula_pdf_from_PITs(u, theta)
kl_clayton = np.array([full_kl(u_big, pdf_sGumbel, pdf_Clayton[i]) for i in range(len(theta_J))])

plt.figure(figsize=(6, 4))
plt.plot(theta_J, kl_clayton, marker='o', linestyle='-')
plt.xlabel(r"theta_{Clayton}")
plt.ylabel("KL divergence to true (sGumbel)")
plt.title(r"KL distances: theta_{Clayton} vs KL distance")
plt.grid(True)
plt.tight_layout()
plt.show()