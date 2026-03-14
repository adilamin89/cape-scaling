"""Diagnostic utilities for symbolic scaling analysis.

This module provides functions to compute the effective scaling exponent,
curvature, divergence operator and collapsed variables from arrays of
parameter counts and losses.  It is intended to accompany the
``Symbolic Scaling Loss'' paper.
"""
from typing import Dict, Any
import numpy as np


def diagnostics(N: np.ndarray, L: np.ndarray) -> Dict[str, Any]:
    """Compute diagnostic quantities for scaling analysis.

    Parameters
    ----------
    N : np.ndarray
        Array of effective parameter counts (must be positive and strictly increasing).
    L : np.ndarray
        Array of corresponding losses (must be positive).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing arrays for the effective exponent, curvature,
        divergence operator, an estimate of the critical scale, and collapsed variables.
    """
    N = np.asarray(N, dtype=float)
    L = np.asarray(L, dtype=float)
    x = np.log(N)
    y = np.log(np.maximum(L, 1e-12))
    # First and second derivatives using centred differences
    y1 = np.gradient(y, x)
    y2 = np.gradient(y1, x)
    alpha_eff = -y1
    kappa = y2
    # Example beta values; these may be fitted locally in practice
    beta1, beta0 = 1.0, 0.0
    D_L = y2 + beta1 * y1 + beta0
    # Estimate critical scale as the point where curvature crosses zero
    idx_c = int(np.argmin(np.abs(kappa)))
    N_c = float(N[idx_c])
    # Heuristic collapse parameters (can be learned from data)
    L_inf = float(np.percentile(L, 5))
    alpha_crit, mu = 0.2, 0.5
    Lstar = (L - L_inf) * (N**alpha_crit)
    xstar = N**(-mu)
    return {
        "alpha_eff": alpha_eff,
        "kappa": kappa,
        "D_L": D_L,
        "N_c": N_c,
        "Lstar": Lstar,
        "xstar": xstar,
    }
