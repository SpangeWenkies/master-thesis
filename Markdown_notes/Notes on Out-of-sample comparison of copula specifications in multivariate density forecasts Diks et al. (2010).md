- **Copula selection problem from an out-of-sample perspective**
  - → Compare predictive accuracy using copula density forecasts and their out-of-sample log-likelihood scores
  - Difference in log-likelihood score for competing density forecasts = their **relative KLIC values**

- This test of copula predictive accuracy can be applied to fully/semi/non-parametric copula-based multivariate density models

 Like in *Giacomini & White (2006)*, unknown parameters estimated in a rolling window have:
- Setups used to test computing density forecasts based on different copula specifications of two competing forecast methods
  - → Model comparison simplifies test procedure, as parameter estimation uncertainty plays no part
  - → It is simply part of the forecasting method

- Examine size/power of these copula predictive accuracy tests via MC
  - Framework for this: **SCOMDY**
  - Resulting test satisfactory in terms of size/power properties

- **Sklar's Theorem**: joint dist. $F(y)$ of $Y_t$ can be expressed in terms of marginals & copula function, i.e.  
  $$
  F(y) = C(F_1(y_1), F_2(y_2), ..., F_d(y_d))
  $$

---

## Conditional Copulas and Patton (2006)

- *Patton (2006)* extends Sklar’s Th. to conditional distributions:  
  $$
  F(y_t | \mathcal{F}_{t-1}) = C(F_1(y_{1,t} | \mathcal{F}_{t-1}), ..., F_d(y_{d,t} | \mathcal{F}_{t-1}) | \mathcal{F}_{t-1})
  $$

- If we define $\tilde{C}$ as the survival copula for $C$, we can define upper and lower tail dependence coefficients:
  $$
  \lambda_L = \lim_{q \to 0} \frac{C(q,...,q)}{q}, \quad \lambda_U = \lim_{q \to 0} \frac{\tilde{C}(q,...,q)}{q}
  $$

- Advantage of fully parametric models: ML can be used to obtain estimates (efficient)

- Log-likelihood for ML estimation (includes copula part):
  $$
  LL = \sum_{j=1}^d \log f_j(y_{j,t} | \mathcal{F}_{t-1}) + \log c(F_1(y_{1,t} | \mathcal{F}_{t-1}), ..., F_d(y_{d,t} | \mathcal{F}_{t-1}) | \mathcal{F}_{t-1})
  $$

- Full copula density:
  $$
  c(u_1, ..., u_d | \mathcal{F}_{t-1}) = \frac{\partial^d}{\partial u_1 ... \partial u_d} C(u_1, ..., u_d | \mathcal{F}_{t-1})
  $$

---

## Estimation Methods

- Conditional copulas and marginals can be estimated simultaneously by maximizing LL based on $T$ observations.

- If parameters can be separated, a **2-stage procedure** is possible:
  - **Inference Functions for Margins (IFM)** method:
    1. Maximize $\log f_j(y_{j,t} | \mathcal{F}_{t-1})$ univariately by ML
    2. Estimate copula parameters by maximizing $\log c(F_1(...), ..., F_d(...) | \mathcal{F}_{t-1})$ conditional on step 1 estimates

- Fully parametric approach assumes correct marginal & copula specification  
  → If not correct, **severe bias in copula parameter estimates**  
  ← *Fermanian & Scaillet (2005)*

---

## Semi-parametric & Sieve Estimation

- *Chen et al. (2006)*: Sieve estimates are more efficient than using ECDF  
  → Also mentioned in *Medovikov et al. (2025)*, important paper to read

### SCOMDY Framework:
- Conditional mean of $Y_t$ — **parametric**
- Conditional variance of $Y_t$ — **parametric**
- Innovations — **semi-parametric**
-> Marginals non-parametric, copula parametric

- General model:
  $$
  Y_t = \mu_t(\theta_1) + \sqrt{H_t(\theta)} \cdot \varepsilon_t
  $$

  where:
  - $\mu_t(\theta_1) = (\mu_{1,t}(\theta_1), ..., \mu_{d,t}(\theta_1))'=E[Y_t | \mathcal{F}_{t-1}]$  
  - $H_t(\theta) = \text{diag}(h_{1,t}(\theta), ..., h_{d,t}(\theta))$  
  - $h_{j,t}(\theta) = h_j(t, \theta_1, \theta_2) = E[(Y_{j,t} - \mu_{j,t}(\theta_1))^2 | \mathcal{F}_{t-1}]$

- $\varepsilon_t$ is i.i.d. with $E[\varepsilon_{j,t}] = 0$, $E[\varepsilon_{j,t}^2] = 1$

- From Sklar's Theorem, joint dist. of $\varepsilon_t$:
  $$
  F(\varepsilon) = C(F_1(\varepsilon_{1}), ..., F_d(\varepsilon_{d}); \alpha)\equiv C(u_1,...,u_d ; \alpha), \quad C: [0,1]^d \to [0,1]
  $$

---

## Chen & Fan 3-Step Estimation Procedure

1. Univariate QML under assumption of normality of $\varepsilon_{j,t}$ to estimate $\hat{\theta}_1$, $\hat{\theta}_2$
2. Marginal distribution estimated via **ECDF** of residuals:  
   $$
   \hat{\varepsilon}_{j,t} \equiv (y_{j,t} - \mu_{j,t}(\hat{\theta}_1)) / \sqrt{h_{j,t}(\hat{\theta})}
   $$
3. Copula parameters estimated by maximizing corresponding **copula log-likelihood**

---

## Uncertainty in Copula Form

- When there is **large uncertainty** in the copula’s functional form:
  - → Use **machine learning / image recognition** (new)
  - → Use **non-parametric copula estimation**  
    (usually subject to **curse of dimensionality**)

---

## Equal Predictive Accuracy Tests for Copulas
- [Placeholder for related discussion]
