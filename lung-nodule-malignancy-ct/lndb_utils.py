"""
lndb_utils.py — Shared configuration and helpers for all project notebooks.

Lab IA & CD – Project 1 | Lung nodule malignancy on CT (LNDb + LIDC-IDRI)

Why a shared module?
    Every notebook needs the same paths, the same class definitions and the same
    file-lookup logic. Defining them once here means a change (e.g. a folder name)
    is made in ONE place and every notebook stays consistent.

Expected folder layout (relative to this file):

    projeto_labiacd/
    ├── lndb_utils.py            ← this file
    ├── data/data0 ... data5/    ← training CT scans (LNDb-XXXX.mhd / .raw)
    ├── masks/                   ← radiologist segmentations (LNDb-XXXX_radR.mhd / .raw)
    ├── testdata0/, testdata1/   ← test CT scans
    ├── trainset_csv/            ← trainNodules.csv, trainNodules_gt.csv, ... and, from LNDb v4,
    │                              chars_trainNodules.csv (malignancy), allNods.csv, LNDbAcqParams.csv
    ├── testset_csv/             ← testNodules.csv
    ├── LIDC-IDRI/               ← LIDC-IDRI scans (download_lidc.py)
    └── outputs/                 ← everything the notebooks generate (created automatically)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 1. Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent
TRAIN_CSV_DIR = ROOT / 'trainset_csv'
TEST_CSV_DIR  = ROOT / 'testset_csv'
MASK_DIR      = ROOT / 'masks'
OUTPUT_DIR    = ROOT / 'outputs'

# CT scans are split across several folders (data0..data5, testdata0..testdata1).
# We search them all, recursively, so the exact sub-folder structure does not matter.
TRAIN_IMAGE_DIRS = sorted((ROOT / 'data').glob('data*'))
TEST_IMAGE_DIRS  = sorted(ROOT.glob('testdata*'))

# LIDC-IDRI scans downloaded by download_lidc.py: LIDC-IDRI/<patient>/<study_uid>/<series_uid>/*.dcm
LIDC_DIR = ROOT / 'LIDC-IDRI'

OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2a. Target definition — nodule MALIGNANCY (the competition's target)
# ─────────────────────────────────────────────────────────────────────────────
# Every LNDb radiologist rated each nodule's likelihood of malignancy on the LIDC-IDRI scale:
#   1 highly unlikely · 2 moderately unlikely · 3 indeterminate · 4 moderately suspicious · 5 highly suspicious
# The ratings are in chars_trainNodules.csv (LNDb v3/v4 on Zenodo), one row per row of
# trainNodules.csv. A nodule's label is the AVERAGE of its radiologists' ratings; the class is
# that average rounded half-up (3.5 → 4), the same rule used for LIDC-IDRI.
MAL_THRESHOLDS = (1.5, 2.5, 3.5, 4.5)
MAL_LABELS     = np.arange(1, 6)
MALIGNANCY_COLORS = ['#1a9850', '#91cf60', '#fee08b', '#fc8d59', '#d73027']   # 1 → 5, green → red
CHARACTERISTICS = ['calcification', 'internalStructure', 'lobulation', 'malignancy', 'margin',
                   'sphericity', 'spiculation', 'subtlety']


def malignancy_to_class(mean_rating: float) -> int:
    """Average malignancy rating (1–5, possibly fractional) → integer class 1–5 (half-up)."""
    return int(np.clip(np.floor(mean_rating + 0.5), 1, 5))


# ─────────────────────────────────────────────────────────────────────────────
# 2b. Texture — clinical context and the project's first model
# ─────────────────────────────────────────────────────────────────────────────
# The `Text` column in LNDb is the radiologists' texture rating (1–5), NOT malignancy.
#   1–2 → ground-glass opacity (GGO)
#   3   → part-solid
#   4–5 → solid
# In trainNodules_gt.csv `Text` is the AVERAGE rating of the radiologists who saw the
# nodule (e.g. 4.33). The official LNDb challenge maps that average to 3 classes with
# these thresholds (https://lndb.grand-challenge.org/Evaluation/):
#   average < 2.33        → GGO
#   2.33 ≤ average ≤ 3.67 → part-solid
#   average > 3.67        → solid
TEXTURE_LOW_THRESHOLD  = 7 / 3    # 2.333...
TEXTURE_HIGH_THRESHOLD = 11 / 3   # 3.666...

CLASS_NAMES = ['GGO', 'Part-solid', 'Solid']   # index = class id (0, 1, 2)

# The competition asks for an integer 1–5 per finding. Each clinical class is
# exported as its representative rating on the original scale.
CLASS_TO_SCORE = {0: 1, 1: 3, 2: 5}

# Nodules < 3 mm were only marked with a centroid (no segmentation). LNDb assigns
# them the volume of a 3 mm diameter sphere: 4/3·π·1.5³ ≈ 14.137 mm³.
SMALL_NODULE_VOLUME = 4 / 3 * np.pi * 1.5 ** 3

# ─────────────────────────────────────────────────────────────────────────────
# 3. Plot style — one palette for every notebook
# ─────────────────────────────────────────────────────────────────────────────
CLASS_COLORS  = {'GGO': '#7FB3D5', 'Part-solid': '#F5B041', 'Solid': '#C0392B'}
COLOR_NEUTRAL = '#5B5EA6'

# Where a feature comes from (Notebooks 3–4). Fixed order, validated for colour-blind
# separation; always shown with a legend or labels, never colour alone.
SOURCE_COLORS = {'original': '#2a78d6', 'wavelet': '#eb6834', 'LoG': '#1baf7a', 'composition': '#eda100'}
COLOR_ACCENT  = '#C0392B'

PLOT_STYLE = {
    'figure.dpi'       : 110,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'font.size'        : 11,
}

RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
# 4. Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def texture_to_class(avg_texture: float) -> int:
    """Map an (averaged) LNDb texture rating 1–5 to a clinical class id 0/1/2."""
    if avg_texture < TEXTURE_LOW_THRESHOLD:
        return 0
    if avg_texture <= TEXTURE_HIGH_THRESHOLD:
        return 1
    return 2


_CT_NAME = re.compile(r'^LNDb-(\d{4})\.mhd$')


def mhd_is_complete(mhd: Path) -> bool:
    """True if the .mhd header AND the data file it points to (usually the .raw) both exist.

    A .mhd is only a small text header; the voxels live in the file named on its
    `ElementDataFile` line. An interrupted archive extraction can leave the header
    without its data file.
    """
    try:
        for line in mhd.read_text(errors='ignore').splitlines():
            if line.strip().startswith('ElementDataFile'):
                name = line.split('=', 1)[1].strip()
                return name.upper() == 'LOCAL' or (mhd.parent / name).exists()
    except OSError:
        return False
    return False


def build_ct_index(image_dirs: list[Path], report_incomplete: bool = True) -> dict[int, Path]:
    """
    Return {LNDbID: path to .mhd} for every complete CT found in `image_dirs` (recursively).

    Masks (LNDb-XXXX_radR.mhd) are ignored because they do not match the CT name pattern.
    CTs whose .raw data file is missing are left out (and reported).
    """
    index, incomplete = {}, []
    for folder in image_dirs:
        for path in folder.rglob('LNDb-*.mhd'):
            match = _CT_NAME.match(path.name)
            if match:
                if mhd_is_complete(path):
                    index[int(match.group(1))] = path
                else:
                    incomplete.append(path.name)
    if incomplete and report_incomplete:
        print(f'⚠️  {len(incomplete)} CT header(s) without their .raw data file (skipped): '
              f'{sorted(incomplete)}')
    return index


def mask_path(lndb_id: int, rad_id: int) -> Path:
    """Path of the segmentation mask drawn by radiologist `rad_id` on CT `lndb_id`.

    Each mask is a 3D volume the size of the CT where the value of every voxel is the
    radiologist's FindingID for that nodule (0 = background).
    """
    return MASK_DIR / f'LNDb-{lndb_id:04d}_rad{rad_id}.mhd'


def load_fold_map() -> dict[int, int]:
    """Return {LNDbID: fold} from the official 4-fold split in trainFolds.csv."""
    folds = pd.read_csv(TRAIN_CSV_DIR / 'trainFolds.csv')
    fold_map = {}
    for fold_idx, col in enumerate(folds.columns):
        for lndb_id in folds[col].dropna().astype(int):
            fold_map[lndb_id] = fold_idx
    return fold_map


def load_ratings_per_radiologist() -> pd.DataFrame:
    """trainNodules.csv with each radiologist's characteristics (chars_trainNodules.csv) attached.

    The two files have the same rows in the same order (same LNDbID and FindingID on every row).
    Only `RadID` differs: trainNodules.csv numbers the radiologists of each CT 1, 2, 3, while
    chars_trainNodules.csv uses the radiologist's identity across the dataset (1–5). The rows are
    therefore joined by position, after checking that CT and finding agree on every row; the global
    identity is kept as `RadGlobal`. A rating of 0 means "not rated" and becomes NaN.
    """
    nodules = pd.read_csv(TRAIN_CSV_DIR / 'trainNodules.csv')
    chars = pd.read_csv(TRAIN_CSV_DIR / 'chars_trainNodules.csv')
    assert len(nodules) == len(chars)
    assert (nodules[['LNDbID', 'FindingID']].values == chars[['LNDbID', 'FindingID']].astype(int).values).all()
    out = nodules.copy()
    out['RadGlobal'] = chars['RadID'].astype(int).values
    for c in CHARACTERISTICS:
        out[c] = chars[c].replace(0, np.nan).values
    return out


def load_malignancy_labels() -> pd.DataFrame:
    """One row per LNDb nodule (Nodule == 1 in trainNodules_gt.csv) with its malignancy label.

    Columns: LNDbID, FindingID, n_readers, readers (global radiologist ids), malignancy_ratings
    ("2,3"), malignancy_mean, malignancy_sd, malignancy_class (1–5) and the average of every other
    characteristic (`<name>_mean`). Nodules no radiologist rated for malignancy get NaN.
    """
    per = load_ratings_per_radiologist().set_index(['LNDbID', 'RadID', 'FindingID'])
    gt = pd.read_csv(TRAIN_CSV_DIR / 'trainNodules_gt.csv')
    rows = []
    for _, r in gt[gt.Nodule == 1].iterrows():
        keys = [(r.LNDbID, int(a), int(b)) for a, b in zip(str(r.RadID).split(','), str(r.RadFindingID).split(','))]
        sub = per.loc[keys]
        m = sub['malignancy'].dropna().values
        rows.append({'LNDbID': r.LNDbID, 'FindingID': r.FindingID, 'n_readers': len(keys),
                     'readers': ','.join(map(str, sub['RadGlobal'])),
                     'malignancy_ratings': ','.join(str(int(v)) for v in m),
                     'malignancy_mean': m.mean() if len(m) else np.nan,
                     'malignancy_sd': m.std() if len(m) else np.nan,
                     **{f'{c}_mean': sub[c].mean() for c in CHARACTERISTICS if c != 'malignancy'}})
    labels = pd.DataFrame(rows)
    labels['malignancy_class'] = labels['malignancy_mean'].map(
        lambda v: malignancy_to_class(v) if v == v else np.nan)
    return labels
