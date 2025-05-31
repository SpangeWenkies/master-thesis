## General Assessment
- Procedure to evaluate predictive risk for limited regions of outcome space.
- Procedure has advantages over others and nice properties.
- Key insight is applying scoring rule to censored distributions.

**However:**
- More closely compare to existing methods and go more in-depth empirically.

---

## Points & Answers

### 2. Look at procedure vs. all procedures more in depth
- *Mitchell & Weale (2023)* is also based on censoring.

### 2a. Alternatives are:
- *Mitchell & Weale (2023)* — weighted likelihood scoring rules
- *Allen et al. (2023)* — weighted kernel scores
- *Holzmann & Klar (2017a)* — general procedure & conditional likelihood score

### HK:
- As opposed to H&K, no auxiliary scoring rule needs to be introduced.
- *Diks et al.* procedure preserves initially chosen divergence measure by researcher.
- In Section 2.2, "sweet spot" clarified in introduction of censoring and "minimal localization" for indicator weight functions.
- Clarification how censoring does not nest, nor is nested in H&K procedure.
  - Example given by *Pelenis (2014)* scoring rule.
  - It is as censoring is guaranteed to deliver localized divergence.
- Now explicit score divergence of H&K provided, revealing auxiliary term which does **not originate from input divergence**, can now *dominate*.
  - In extreme case, even independence of auxiliary & input divergence.
- If auxiliary is KL divergence, then it can always be expected.
- H&K has good power properties, regardless of input divergence choice (due to NP).

---

## MW:
- Aim to perform statistical inference based on some central region, **not evaluating multiple candidate densities**.
- They define two censored log-likelihoods, both regions defined endogenously, i.e., 100 × α% central region of forecast density of minimal length.
- In *Diks et al. (2023)*, this does not work as the scoring rule for comparing candidate distributions is then rendered **improper**.
  - Improper scoring rule → **spurious power** (*Diks et al. 2011*).
  - From the above, *Diks et al. (2023)* include example of improperness in their paper (revised version).

---

## On Model Rankings:
- MCS used to evaluate performance of *Diks et al.*, H&K, *Allen et al.*
- ***Sign added to show MCS relevance to Diks et al. rules***
  - MCS becomes smaller using these correlations — consistent with the auxiliary rule’s beneficial power properties.

---

## Scope Broadened:
- Inclusion of **non-indicator weight functions**.
- Better for some empirical applications.
- More in line with *Allen et al. (2023)* who develop localized scoring rules for class of kernel scores.
