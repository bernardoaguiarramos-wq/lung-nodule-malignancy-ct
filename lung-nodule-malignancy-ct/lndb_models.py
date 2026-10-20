"""
lndb_models.py — Model definitions and the ordinal "texture score committee".

Lab IA & CD – Project 1 | Lung nodule malignancy on CT (LNDb + LIDC-IDRI)

Two ways of predicting texture are defined here and compared in Notebook 4:

A. CLASSIFIERS (the classic approach)
       features ──► KNN / SVM / MLP ──► GGO | part-solid | solid
   Every pipeline tunes how many features to use (selector inside the pipeline, no leakage).

B. TEXTURE SCORE COMMITTEE (our approach)
       features ──► KNN, SVM, MLP regressors ──► average score (1–5) ──► learned cut-offs ──► class
   Instead of learning three buckets, each model learns the radiologists' AVERAGE texture rating
   (e.g. 3.67). That number keeps information the buckets throw away (a 3.67 is "almost solid",
   a 2.4 "almost GGO") and respects the order GGO < part-solid < solid. The three models vote by
   averaging their scores — like the dataset itself averages its radiologists — and the two
   cut-offs that turn the score into a class are set from out-of-fold predictions so that each
   class is predicted as often as radiologists assign it (prevalence matching).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

import lndb_utils as U
from lndb_selection import RelevanceRedundancySelector

# Default feature-selection settings. Notebook 4 overrides them with the values Notebook 3 saved in
# outputs/selected_features.json, so the two notebooks always use the same settings.
DEFAULT_SELECTION = {'n_features_candidates': [5, 10, 20, 50, 100, 'all'],
                     'corr_threshold': 0.85, 'fdr_alpha': 0.05}


def selection_settings(selection: dict | None = None) -> dict:
    """Merge user-provided selection settings (e.g. Notebook 3's JSON) with the defaults."""
    out = dict(DEFAULT_SELECTION)
    if selection:
        out.update({k: selection[k] for k in DEFAULT_SELECTION if k in selection})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def qwk(y_true, y_pred) -> float:
    """Quadratic weighted kappa — the official LNDb texture metric."""
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


TEXTURE_THRESHOLDS = (U.TEXTURE_LOW_THRESHOLD, U.TEXTURE_HIGH_THRESHOLD)


def score_to_class(score, thresholds=TEXTURE_THRESHOLDS, half_up: bool = False) -> np.ndarray:
    """Turn a continuous score into a class id 0 … K-1 with K-1 increasing cut-offs.

    Ties exactly on a cut-off follow the rule of the target:
    * texture (default, `half_up=False`) — the official LNDb rule: GGO if score < low,
      part-solid if low ≤ score ≤ high, solid if score > high;
    * malignancy (`half_up=True`) — ordinary rounding: a score equal to a cut-off goes up.
    """
    score = np.asarray(score, dtype=float)
    t = np.asarray(thresholds, dtype=float)
    if half_up:
        return (score[:, None] >= t[None, :]).sum(axis=1)
    return (score >= t[0]).astype(int) + (score[:, None] > t[None, 1:]).sum(axis=1)


def _pearson(y_true, y_pred) -> float:
    if np.ptp(y_pred) == 0:
        return 0.0
    return pearsonr(y_true, y_pred)[0]


QWK_SCORER = make_scorer(qwk)
# For the regressors we tune hyper-parameters on how well the score RANKS the nodules
# (correlation); the cut-offs are learned afterwards, so the absolute scale does not matter.
CORR_SCORER = make_scorer(_pearson)


def describe_params(best_params: dict) -> dict:
    """Readable version of a grid-search result: model hyper-parameters + number of features."""
    out = {k.split('__', 1)[1]: v for k, v in best_params.items() if k.startswith('model__')}
    out['n_features'] = best_params.get('firstk__k', 'all')
    return out


def _fast_qwk(y, pred) -> float:
    """QWK for 3 classes from a confusion matrix — same result as cohen_kappa_score, ~50× faster."""
    cm = np.bincount(3 * y + pred, minlength=9).reshape(3, 3).astype(float)
    i, j = np.indices((3, 3))
    w = (i - j) ** 2
    expected = np.outer(cm.sum(1), cm.sum(0)) / cm.sum()
    denom = (w * expected).sum()
    return 1 - (w * cm).sum() / denom if denom > 0 else 0.0


def _best_thresholds(score, y, grid) -> tuple[float, float]:
    """Exhaustive search over every pair low < high of `grid`, vectorised.

    For a pair (low, high) the confusion matrix only needs, per true class, how many scores are
    below `low` and how many are ≤ `high` — cumulative counts computed once for the whole grid.
    """
    lt = np.array([(score[y == c][:, None] < grid).sum(0) for c in range(3)], dtype=float)   # (3, G)
    le = np.array([(score[y == c][:, None] <= grid).sum(0) for c in range(3)], dtype=float)
    n_c = np.bincount(y, minlength=3).astype(float)
    # cm[c, p] for every (low index i, high index j): shape (3, 3, G, G)
    G = len(grid)
    p0 = np.broadcast_to(lt[:, :, None], (3, G, G))                     # predicted GGO: score < low
    p2 = np.broadcast_to(n_c[:, None, None] - le[:, None, :], (3, G, G))  # predicted solid: score > high
    p1 = n_c[:, None, None] - p0 - p2                                    # the rest: part-solid
    cm = np.stack([p0, p1, p2], axis=1)
    w = (np.arange(3)[:, None] - np.arange(3)[None, :]) ** 2
    observed = (w[:, :, None, None] * cm).sum((0, 1))
    expected = (w[:, :, None, None] * n_c[:, None, None, None] * cm.sum(0)[None]).sum((0, 1)) / n_c.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        kappa = np.where(expected > 0, 1 - observed / expected, -np.inf)
    kappa[np.tril_indices(G)] = -np.inf                      # keep only low < high
    i, j = np.unravel_index(np.argmax(kappa), kappa.shape)
    return float(grid[i]), float(grid[j])


def prevalence_thresholds(score, y, n_classes: int = 3) -> tuple:
    """Cut-offs that make the predicted class frequencies equal to the observed ones.

    If 4.7% of training nodules are GGO and 8.2% part-solid, the lowest 4.7% of scores become GGO,
    the next 8.2% part-solid and the rest solid. No metric is optimised, so the cut-offs cannot
    over-fit a handful of rare nodules, and no class can silently disappear.
    """
    share = np.cumsum(np.bincount(np.asarray(y, dtype=int), minlength=n_classes)) / len(y)
    cut = np.nanquantile(np.asarray(score, dtype=float), np.clip(share[:n_classes - 1], 0, 1))
    return tuple(round(float(c), 3) for c in cut)


CUTOFF_STRATEGIES = {
    'official': lambda score, y: (U.TEXTURE_LOW_THRESHOLD, U.TEXTURE_HIGH_THRESHOLD),
    'qwk':      lambda score, y: tune_thresholds(score, y),
    'prevalence': prevalence_thresholds,
}


def tune_thresholds(score, y, n_bootstrap: int = 25, step: float = 0.05,
                    random_state: int = U.RANDOM_STATE) -> tuple[float, float]:
    """Two cut-offs (low < high) that maximise QWK on (score, y) — bagged for stability.

    The single best pair of cut-offs on one sample can be very sensitive to a handful of rare
    nodules. We search the best pair on `n_bootstrap` resamples of the data and keep the MEDIAN
    of each cut-off, which is much more stable from one training set to another.
    """
    score, y = np.asarray(score, dtype=float), np.asarray(y, dtype=int)
    grid = np.round(np.arange(1.0, 5.0 + step, step), 3)
    if not n_bootstrap:
        return _best_thresholds(score, y, grid)
    rng = np.random.default_rng(random_state)
    found = np.array([_best_thresholds(score[idx], y[idx], grid)
                      for idx in (rng.integers(0, len(y), len(y)) for _ in range(n_bootstrap))])
    low, high = np.median(found, axis=0)
    return round(float(low), 3), round(float(max(high, low + step)), 3)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline building blocks
# ─────────────────────────────────────────────────────────────────────────────
class FirstK(BaseEstimator, TransformerMixin):
    """Keep the first k columns.

    The selector orders features from most to least relevant, and its greedy choice of the
    first k features does not depend on k. So we run the (slow) selector ONCE with no limit
    and let this (instant) step try different k — the grid search becomes much faster.
    """

    def __init__(self, k: int | None = None):
        self.k = k

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X)[:, :self.k]


class ScoreSelector(RelevanceRedundancySelector):
    """Same selector, for models trained on the continuous score: relevance is computed
    against the clinical class the score corresponds to."""

    def __init__(self, k=15, corr_threshold=0.85, fdr_alpha=0.05,
                 class_thresholds=TEXTURE_THRESHOLDS, half_up: bool = False):
        super().__init__(k=k, corr_threshold=corr_threshold, fdr_alpha=fdr_alpha)
        self.class_thresholds = class_thresholds
        self.half_up = half_up

    def fit(self, X, y):
        return super().fit(X, score_to_class(y, self.class_thresholds, self.half_up))


def _steps(selector_cls, model, oversample: bool, settings: dict, **selector_args) -> list:
    steps = [
        ('impute',   SimpleImputer(strategy='median')),   # safety net for failed extractions
        ('select',   selector_cls(k=None, corr_threshold=settings['corr_threshold'],
                                  fdr_alpha=settings['fdr_alpha'], **selector_args)),
        ('firstk',   FirstK()),
        ('quantile', QuantileTransformer(n_quantiles=200, output_distribution='normal')),
        ('scale',    StandardScaler()),
    ]
    if oversample:
        steps.append(('smote', SMOTE(random_state=U.RANDOM_STATE)))
    steps.append(('model', model))
    return steps


def _grid(model_params: dict, candidates) -> list[dict]:
    """Grid = model hyper-parameters × number of features.

    `candidates` mixes integers (keep the k best features) and 'all' (no selection at all).
    """
    ks = [k for k in candidates if k != 'all']
    grid = [{'firstk__k': ks, **model_params}] if ks else []
    if 'all' in candidates:
        grid.append({'select': ['passthrough'], 'firstk': ['passthrough'], **model_params})
    return grid


# ─────────────────────────────────────────────────────────────────────────────
# A. Classifiers
# ─────────────────────────────────────────────────────────────────────────────
def make_classifier(name: str, selection: dict | None = None, memory=None):
    """Return (pipeline, parameter grid) for a 3-class texture classifier.

    The number of features is tuned over Notebook 3's candidates (`selection`).
    """
    rs, settings = U.RANDOM_STATE, selection_settings(selection)
    if name == 'KNN':
        model, oversample = KNeighborsClassifier(weights='distance'), True
        params = {'model__n_neighbors': [5, 15, 31]}
    elif name == 'SVM':
        model, oversample = SVC(kernel='rbf', class_weight='balanced'), False
        params = {'model__C': [0.1, 1, 10], 'model__gamma': ['scale', 1e-3]}
    elif name == 'MLP':
        model, oversample = MLPClassifier(max_iter=600, random_state=rs), True
        params = {'model__hidden_layer_sizes': [(32,), (64, 32)], 'model__alpha': [1.0, 10.0]}
    elif name == 'LogReg':
        model, oversample = LogisticRegression(class_weight='balanced', max_iter=4000), False
        params = {'model__C': [0.05, 0.5, 5]}
    else:
        raise ValueError(name)
    pipe = Pipeline(_steps(RelevanceRedundancySelector, model, oversample, settings), memory=memory)
    return pipe, _grid(params, settings['n_features_candidates'])


# ─────────────────────────────────────────────────────────────────────────────
# B. Regressors and the committee
# ─────────────────────────────────────────────────────────────────────────────
REGRESSOR_USES_WEIGHTS = {'KNN': False, 'SVM': True, 'MLP': True}


def make_regressor(name: str, selection: dict | None = None, memory=None,
                   class_thresholds=TEXTURE_THRESHOLDS, half_up: bool = False):
    """Return (pipeline, parameter grid) for a texture-SCORE regressor.

    KNN tunes the number of features over Notebook 3's candidates. SVM and MLP always use every
    feature: as classifiers (Notebook 4, section 5) these regularised models chose 100 features or
    all of them, and fixing 'all' keeps the nested search affordable on a laptop.
    """
    rs, settings = U.RANDOM_STATE, selection_settings(selection)
    if name == 'KNN':
        model = KNeighborsRegressor(weights='distance')
        grid = _grid({'model__n_neighbors': [15, 31]}, settings['n_features_candidates'])
    elif name == 'SVM':
        model = SVR(kernel='rbf')
        grid = _grid({'model__C': [1, 10]}, ['all'])
    elif name == 'MLP':
        model = MLPRegressor(hidden_layer_sizes=(32,), max_iter=800, random_state=rs)
        grid = _grid({'model__alpha': [1.0, 10.0]}, ['all'])
    else:
        raise ValueError(name)
    steps = _steps(ScoreSelector, model, oversample=False, settings=settings,
                   class_thresholds=class_thresholds, half_up=half_up)
    return Pipeline(steps, memory=memory), grid


class TextureScoreCommittee(BaseEstimator, ClassifierMixin):
    """KNN + SVM + MLP regressors on a 1–5 rating, averaged, with cut-offs set from
    out-of-fold predictions.

    Built for texture (Notebook 4) and reused for malignancy (Notebook 6): the target is any
    radiologist rating whose classes are defined by cut-offs on the (averaged) rating.
    `class_thresholds` / `half_up` define those classes — texture by default (2.33 / 3.67, the LNDb
    rule); malignancy uses (1.5, 2.5, 3.5, 4.5) with `half_up=True`, i.e. ordinary rounding.

    fit(X, score, groups)
        1. For each member: grouped grid search of its hyper-parameters (inner CV).
        2. Out-of-fold scores of each tuned member on the same inner splits.
        3. Cut-offs on the averaged out-of-fold score (`cutoffs`: 'prevalence', 'official', or
           'qwk' — the last one only for 3 classes).
        4. Each member is refitted on all the data it was given.
    Rare classes are up-weighted (balanced sample weights) for the members that support it.
    """

    def __init__(self, members=('KNN', 'SVM', 'MLP'), cutoffs: str = 'prevalence', selection=None,
                 class_thresholds=TEXTURE_THRESHOLDS, half_up: bool = False,
                 inner_splits: int = 4, n_jobs: int = -1, random_state: int = U.RANDOM_STATE, memory=None):
        self.members = members
        self.cutoffs = cutoffs
        self.selection = selection
        self.class_thresholds = class_thresholds
        self.half_up = half_up
        self.inner_splits = inner_splits
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.memory = memory

    def fit(self, X, score, groups, sample_weight=None):
        """`sample_weight` (optional): how much each nodule should count, e.g. the reliability of its
        label (Notebook 7). It multiplies the balanced class weights; both are used by the members
        that support weights (SVM, MLP)."""
        X, score = np.asarray(X, dtype=float), np.asarray(score, dtype=float)
        y = score_to_class(score, self.class_thresholds, self.half_up)
        n_classes = len(self.class_thresholds) + 1
        weights = compute_sample_weight('balanced', y)
        if sample_weight is not None:
            weights = weights * np.asarray(sample_weight, dtype=float)
            weights = weights / weights.mean()
        splits = list(StratifiedGroupKFold(self.inner_splits, shuffle=True,
                                           random_state=self.random_state).split(X, y, groups))
        self.models_, self.best_params_, oof = {}, {}, {}
        for name in self.members:
            pipe, grid = make_regressor(name, selection=self.selection, memory=self.memory,
                                        class_thresholds=self.class_thresholds, half_up=self.half_up)
            fit_params = {'model__sample_weight': weights} if REGRESSOR_USES_WEIGHTS[name] else {}
            search = GridSearchCV(pipe, grid, scoring=CORR_SCORER, cv=splits, n_jobs=self.n_jobs)
            search.fit(X, score, **fit_params)
            self.best_params_[name] = describe_params(search.best_params_)
            oof[name] = np.zeros(len(y))
            for tr, va in splits:
                fp = {'model__sample_weight': weights[tr]} if REGRESSOR_USES_WEIGHTS[name] else {}
                oof[name][va] = clone(search.best_estimator_).fit(X[tr], score[tr], **fp).predict(X[va])
            self.models_[name] = search.best_estimator_            # already refitted on all data
        self.oof_scores_, self.oof_classes_ = oof, y
        def rule(s):
            if self.cutoffs == 'prevalence':
                return prevalence_thresholds(s, y, n_classes)
            if self.cutoffs == 'official':
                return tuple(self.class_thresholds)
            if self.cutoffs == 'qwk' and n_classes == 3:
                return tune_thresholds(s, y)
            raise ValueError(f'cut-off strategy {self.cutoffs!r} not available for {n_classes} classes')

        self.thresholds_ = rule(np.mean(list(oof.values()), axis=0))
        # Each member's own cut-offs, so that every member can also be evaluated on its own
        self.member_thresholds_ = {name: rule(s) for name, s in oof.items()}
        self.classes_ = np.arange(n_classes)
        return self

    def member_scores(self, X) -> dict[str, np.ndarray]:
        X = np.asarray(X, dtype=float)
        return {name: m.predict(X) for name, m in self.models_.items()}

    def predict_score(self, X) -> np.ndarray:
        """Committee score on the 1–5 rating scale (before the cut-offs)."""
        return np.mean(list(self.member_scores(X).values()), axis=0)

    def predict(self, X) -> np.ndarray:
        return score_to_class(self.predict_score(X), self.thresholds_, self.half_up)

    def predict_members(self, X) -> dict[str, np.ndarray]:
        """Class predicted by each member alone, with its own cut-offs."""
        return {name: score_to_class(s, self.member_thresholds_[name], self.half_up)
                for name, s in self.member_scores(X).items()}
