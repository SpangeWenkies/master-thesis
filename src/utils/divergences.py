#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
divergences.py

Global, local (Def 2) and localised (Def 3) KL divergences
for continuous copula densities on [0,1]^d.

Local ∝ conditional KL on the ROI.
Localised = KL between the weight-renormalised laws P_w and F_w.
"""

from __future__ import annotations
from typing import Callable, Union
import numpy as np

EPS = 1e-12


# ─────────────────── 0.  Basic helpers ──────────────────────────────────
def kl_vector(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> float:
    """KL on a 1-D pmf."""
    p, q = np.asarray(p), np.asarray(q)
    p = np.clip(p, eps, None); q = np.clip(q, eps, None)
    p /= p.sum(); q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def full_kl(u: np.ndarray,
                   pdf_p: Callable[[np.ndarray], np.ndarray],
                   pdf_q: Callable[[np.ndarray], np.ndarray]) -> float:
    """
    Monte-Carlo KL using *given* sample u ~ P, instead of drawing inside.
    """
    p, q = pdf_p(u), pdf_q(u)
    mask = p > 0
    return float(np.mean(
        np.log(p[mask] / np.clip(q[mask], EPS, None))
    ))


# ─────────────────── 1.  Local KL  (Def 2) ──────────────────────────────
def local_kl(u: np.ndarray,
             pdf_p: Callable[[np.ndarray], np.ndarray],
             pdf_q: Callable[[np.ndarray], np.ndarray],
             roi_mask: np.ndarray) -> float:
    """
    Conditional KL  D(P‖Q | ROI) estimated with samples u ~ P.
    """
    idx = roi_mask.astype(bool)
    if idx.sum() == 0:
        return 0.0
    p, q = pdf_p(u[idx]), pdf_q(u[idx])
    return float(np.mean(np.log(np.clip(p, EPS, None) /
                                np.clip(q, EPS, None))))


# ─────────────────── 2.  Localised KL  (Def 3) ──────────────────────────
def localised_kl(u: np.ndarray,
                 pdf_p: Callable[[np.ndarray], np.ndarray],
                 pdf_q: Callable[[np.ndarray], np.ndarray],
                 roi_mask: np.ndarray) -> float:
    """
    KL(P_w ‖ Q_w) where  P_w  is P re-weighted by w (indicator of ROI).

    Uses the identity:
        KL(P_w‖Q_w) = LocalKL + log(Z_Q / Z_P)
    """
    idx = roi_mask.astype(bool)
    p_vals = pdf_p(u)
    q_vals = pdf_q(u)

    Z_P = idx.mean()                                         # E_P[w]
    Z_Q = np.mean(idx * q_vals / np.clip(p_vals, EPS, None)) # ∫ w f_Q

    if Z_P == 0:
        return 0.0      # no ROI hits in sample ⇒ undefined; return 0

    cond_kl = local_kl(u, pdf_p, pdf_q, roi_mask)
    return cond_kl + np.log(np.clip(Z_Q, EPS, None) / Z_P)
