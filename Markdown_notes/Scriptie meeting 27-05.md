## Starting Setup
- Start with choosing DGP for $u_1$, $u_2$ (some copula $C_{true}$). Create candidate density forecasts $C_1$, $C_2$
  - On these we can apply scoring rules like log score, i.e. $S_i$
  - Start by making distribution of $S_1$ for $C_1$ (histogram / kernel density estimator)
  - Now change $u_i$ for $C_1$
    - How: rank them → if some $u_1$ has value $i$, then its rank will be $i / (T + 1)$ or maybe $i / (T + 2)$ to keep from borders where copula density can be infinite
  - From these $\hat{u}_i$, make another histogram / kernel density estimator of the scores
  - Compare these scoring distributions  
    → maybe pairwise? Maybe not insightful/applicable here

---

## Copula Comparison Setup
- What if we look at both copulas $C_1$ and $C_2$?
- What then comes into play is we want to test $H_0$:  
  $$
  \mathbb{E}[S_i(C_1(u))] = \mathbb{E}[S_i(C_2(u))]
  $$
	  - Find example of $C_1$, $C_2$ for which $H_0$ true  
	    → use symmetry or ...
	  - Also find example where $H_0$ is false, like $C_1 = C_{\text{true}}, C_2 \neq C_{\text{true}}$
	  - Now we can do size & power tests, create histogram of score differences under $H_0$ and under $H_a$

- When now $\hat{u}$ is used, do score difference histograms change?  
  → Effect is probably very dependent on sample size
	- Also, what happens to testing?
	- Maybe look at DM-stat (or Wilcoxon rank stat)
	- Now look at size/power 
		- Choose it using asymptotic normal (most commonly used) or t-dist for test statistic
	    - Look at left tail / right tail / two-tailed
	      - Sometimes you reject too much in one of the two regions (one-sided & two-sided testing)

---

## Expectations & Simulation

- What do we expect will happen?
  - Less power due to uncertainty
  - No bias in symmetric copula choice  
    (→ define symmetry in thesis)
  - Possible bias when KL divergence is equal but copulas are not symmetric  
    → important addition in thesis
	- We can find such cases numerically, use tuning parameter and tune until KL divergences are equal  
	  → Not done earlier by *Diks et al.*
	- Simulate $10^6$ points from dist. & calculate average log score → from this get KL divergence and tune
	- Then simulate $10^3$ points and still use that tuning parameter  
	→ Also look at more options in my notes on this

---

## Additional Uncertainty
- Another uncertainty can be introduced by estimating conditional mean & conditional variance — GARCH
---
## What is the $H_0$ implied
- An important question in my paper: **which $H_0$ am I actually testing?**  
  When using $\hat{u}$ in the null, the expectation is then taken also over the estimation procedure

---

## -> Bias in Testing Scoring Rules

- Implications of bias introduced by using $\hat{u}$ instead of $u$:
	- Then in testing a scoring rule, you're **not testing** the null of equal predictive ability you aim to test
	- Size & power tests are then flawed & scoring rule might not be as strong as you think it is
	- Conclusions drawn based on scoring rule are not accurate or true
	- VaR & ES have model selection step → if scoring rule used in that step is flawed <- the impact on VaR & ES can be simulated or maybe looked at empirically?
	- VaR & ES might then be underestimated

---

## Exploration of VaR/ES Bias
- When we want to look at VaR/ES implications empirically, we do not know the real $u_i$  
  → Just stick here to simulating a GARCH process to look at VaR & ES implications

- Empirically we could apply findings by introducing a **penalty** in the method  
  → Find out how to quantify this penalty (e.g. adding $\frac{1}{2}\sigma^2$ to score)

- Could be that models with larger variance would have preference through the bias → then penalize those
  1. First simulate normal vs penalized size/power
  2. Then look empirically

---

## Localization & Methodology Notes

- In our case through **localization**, the regions of interest also come to dictate the null hypothesis → they influence KL divergences
  - Localized case must be viewed as an **extension** to thesis (not the base of it)  
    → Although maybe needed for VaR & ES

---

## Thesis Planning

- Create a **methodology section**, then optionally an empirical section  
  → Methodology will be evaluated next thesis meeting

---

## Localized Extension

- While we evaluate uncertainty introduced by **ECDF restriction / transformation using indicator weight functions**,  
  → In practice, researchers will probably **smooth** the weight function

- Look at effect of **smooth weights vs indicator**, as indicator probably suffers most from uncertainty in $\hat{u}$ in ECDF transformation
- Right now simulation results show:
    - ECDF case: score differences have high variance  
    - Oracle case: score differences have lower variance

      → Higher $\sigma \Rightarrow$ lower test-stat $Q \Rightarrow$ less power

- Recall: region transformation may not depend on the candidates $C_1$ or $C_2$
	- If it does → scores compared on different outcome space regions  
	    → scoring rule rendered **improper**
	- *Mitchell & Weale (2023)* use scoring rules that localize this way
	- In the **revision of Diks et al. (2025)**, the problem via the M&W example is explained
