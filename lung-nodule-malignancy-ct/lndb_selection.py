"""
lndb_selection.py — Feature selection as a reusable scikit-learn step.

Lab IA & CD – Project 1 | Lung nodule malignancy on CT (LNDb + LIDC-IDRI)

Why a scikit-learn transformer?
    Feature selection must be LEARNED FROM THE TRAINING DATA ONLY. If we chose the features
    using all nodules and then evaluated a model with cross-validation, the validation nodules
    would already have influenced which features were kept — an optimistic, leaked estimate.
    Wrapping the selection in a transformer lets it sit inside a Pipeline, so in every
    cross-validation fold it is re-fitted on that fold's training part only.

The selector combines two filters, in this order:

    1. RELEVANCE  — Kruskal–Wallis test per feature (does its distribution differ between the
                    classes — the 5 malignancy classes, or the 3 texture classes?). p-values are corrected for testing ~1136 features
                    at once with the Benjamini–Hochberg false discovery rate (FDR).
    2. REDUNDANCY — Features are visited from most to least relevant; a feature is kept only if
                    its |Spearman correlation| with every already-kept feature is below a
                    threshold. Stops after `k` features.

Both statistics are rank-based, so they are robust to the heavy skew of radiomic features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kruskal, false_discovery_control, rankdata
from sklearn.base import BaseEstimator, TransformerMixin


class RelevanceRedundancySelector(BaseEstimator, TransformerMixin):
    """Keep the `k` most relevant, mutually non-redundant features.

    Parameters
    ----------
    k : int
        Maximum number of features to keep.
    corr_threshold : float
        Two features with |Spearman ρ| above this value are considered redundant.
    fdr_alpha : float
        Features whose FDR-corrected Kruskal–Wallis p-value is above this are discarded.
    """

    def __init__(self, k: int = 15, corr_threshold: float = 0.85, fdr_alpha: float = 0.05):
        self.k = k
        self.corr_threshold = corr_threshold
        self.fdr_alpha = fdr_alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        classes = np.unique(y)

        # 1. Relevance — Kruskal–Wallis H statistic and FDR-corrected p-value per feature
        h, p = np.zeros(X.shape[1]), np.ones(X.shape[1])
        for j in range(X.shape[1]):
            groups = [X[y == c, j] for c in classes]
            if np.ptp(X[:, j]) > 0:
                h[j], p[j] = kruskal(*groups)
        q = false_discovery_control(p)
        self.h_, self.q_ = h, q

        # 2. Redundancy — greedy, from most to least relevant
        ranks = np.apply_along_axis(rankdata, 0, X)
        ranks = (ranks - ranks.mean(0)) / (ranks.std(0) + 1e-12)
        order = [j for j in np.argsort(-h) if q[j] < self.fdr_alpha]
        self.n_relevant_ = len(order)
        if not order:
            # No feature passes the FDR filter (possible with a tiny feature pool or a small
            # training fold). Fall back to ranking all non-constant features by H, so the
            # pipeline never receives an empty feature matrix.
            order = [j for j in np.argsort(-h) if h[j] > 0] or [int(np.argmax(h))]
        selected = []
        for j in order:
            if len(selected) == self.k:
                break
            if selected:
                rho = np.abs(ranks[:, selected].T @ ranks[:, j]) / len(ranks)   # Spearman ρ
                if rho.max() >= self.corr_threshold:
                    continue
            selected.append(j)
        self.selected_ = np.array(selected, dtype=int)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.selected_]

    def selected_names(self, feature_names) -> list[str]:
        return [feature_names[j] for j in self.selected_]


def image_type(feature: str) -> str:
    """Image version a feature comes from: 'original', 'wavelet', 'LoG' or 'composition'."""
    if feature.startswith('comp_'):
        return 'composition'
    kind = feature.split('_')[1]
    return 'wavelet' if kind.startswith('wavelet') else 'LoG' if kind.startswith('log') else 'original'


def readable(feature: str) -> str:
    """Shorter, human-readable feature name for plots."""
    return (feature.replace('rad_', '').replace('log-sigma-', 'LoG σ')
            .replace('-0-mm-3D', 'mm').replace('original_', 'orig_'))
