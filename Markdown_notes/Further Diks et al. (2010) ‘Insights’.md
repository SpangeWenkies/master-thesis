1. **PITs estimated using residuals from (estimated) conditional models**
   - Then converted into ranks over a window to get:

     $Û_{j,t+1} = \frac{\text{rank}(\varepsilon_{j,t+1} | F_t)}{(R + 2)}$  ← see eq. (10) in Diks et al.

   - From these, they compute log-copula scores: $\log \hat{c}_t(Û_{t+1})$ to use

---

2. **In the calculation of the test statistic they treat estimated PITs $Û_{j,t+1}$ as if they were the true PITs**
   - $Q_{R,P} = \sqrt P · \frac{\bar{d}_{R,P}}{σ̂_{R,P}}$
   - Is actually $\hat{Q}_{R,P} = \sqrt P · \frac{\hat{\bar{d}}_{R,P}}{σ̂_{R,P}}$
   - Used to compute scores & estimate copula parameters

   > Implicitly done in sim. setup where ECDF is used to map the residuals to uniforms

---

3. **Size & power tests were conducted under assumption that uncertainties like these are just part of each "forecasting method"**
   - → any bias would just be part of the approach  
     → why is this not a problem?  
       → HAC also introduced uncertainty in test-stat.

   - → They do this as Giacomini & White (2006) did in their framework

   - → If we shift the objective from simply forecast comparison to diagnostics/inference — we should separate model from estimation techniques

**→ We want to look at bias & variability**
