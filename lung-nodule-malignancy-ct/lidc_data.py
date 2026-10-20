"""
lidc_data.py — LIDC-IDRI annotations and feature extraction, compatible with the LNDb pipeline.

Lab IA & CD – Project 1 | Lung nodule malignancy on CT (LNDb + LIDC-IDRI)

Why this module?
    LIDC-IDRI has malignancy ratings from up to four radiologists per nodule, on the same 1–5
    scale as LNDb. It is used as extra training data for the LNDb malignancy model (Notebook 6).
    That only works if a nodule is described in exactly the same way in both datasets, so this
    module reuses the LNDb functions (crop → automatic segmentation → 1136 features) unchanged.

Two sources, two roles
    * Annotations come from the `pylidc` library, which ships the full LIDC annotation database
      (ratings + outlines). They need no image files.
    * Images are the DICOM files downloaded by download_lidc.py, read with SimpleITK.

Coordinates
    pylidc gives a nodule centre as (i, j, k) = (row, column, slice index in pylidc's sorted list).
    Row and column index the same in-plane grid as the DICOM images. The slice index is converted
    to a physical z position with the slice positions stored in pylidc, and then to the slice index
    of the image we read — so the two never need to have been sorted the same way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# pylidc (last release 2020) still uses aliases that NumPy removed in version 1.24.
import warnings                                  # noqa: E402
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for _name, _type in [('int', int), ('float', float), ('bool', bool)]:
        if not hasattr(np, _name):
            setattr(np, _name, _type)

import pylidc as pl                              # noqa: E402
import SimpleITK as sitk                         # noqa: E402
from joblib import Parallel, delayed             # noqa: E402
from pylidc.utils import consensus               # noqa: E402

import lndb_utils as U                           # noqa: E402
import lndb_features as F                        # noqa: E402


def malignancy_to_class(mean_rating: float) -> int:
    """Average malignancy rating (1–5, possibly fractional) → integer class 1–5 (nearest)."""
    return int(np.clip(np.floor(mean_rating + 0.5), 1, 5))


_SERIES_INDEX: dict | None = None


def _series_index() -> dict:
    """SeriesInstanceUID → folder, for LIDC copies NOT laid out as patient/study_uid/series_uid.

    download_lidc.py saves every series as LIDC-IDRI/<patient>/<study UID>/<series UID>/. A copy
    downloaded with TCIA's NBIA Data Retriever uses descriptive folder names instead
    (e.g. LIDC-IDRI-0001/01-01-2000-NA-NA-30178/3000566.000000-NA-03192/). To support both, the
    series UID is read from the first DICOM file of every folder (once per session, a few seconds;
    not saved to disk, so it never goes stale when scans are added or the folder is moved).
    """
    global _SERIES_INDEX
    if _SERIES_INDEX is not None:
        return _SERIES_INDEX
    index = {}
    folders = {p.parent for p in U.LIDC_DIR.rglob('*.dcm')} if U.LIDC_DIR.exists() else set()
    reader = sitk.ImageFileReader()
    for folder in sorted(folders):
        try:
            reader.SetFileName(str(next(folder.glob('*.dcm'))))
            reader.ReadImageInformation()
            index[reader.GetMetaData('0020|000e').strip(' \x00')] = str(folder)
        except Exception:          # unreadable or non-image DICOM: skip
            continue
    _SERIES_INDEX = index
    return index


def series_dir(patient_id: str, study_uid: str, series_uid: str) -> Path:
    """Folder of one CT series: the download_lidc.py layout, or any other layout via the UID index."""
    folder = U.LIDC_DIR / patient_id / study_uid / series_uid
    if folder.exists():
        return folder
    return Path(_series_index().get(series_uid, folder))


def is_downloaded(folder: Path) -> bool:
    return folder.exists() and any(folder.glob('*.dcm'))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Annotations — one row per nodule (no images needed)
# ─────────────────────────────────────────────────────────────────────────────
def nodule_table(only_downloaded: bool = True) -> pd.DataFrame:
    """One row per LIDC nodule: who rated it, its ratings, and its centre in the scan.

    pylidc groups the annotations of the different radiologists that refer to the same nodule
    (`cluster_annotations`). Only nodules ≥ 3 mm are in the database — the same rule as LNDb.
    """
    rows = []
    for scan in pl.query(pl.Scan).all():
        folder = series_dir(scan.patient_id, scan.study_instance_uid, scan.series_instance_uid)
        if only_downloaded and not is_downloaded(folder):
            continue
        try:
            nodules = scan.cluster_annotations(verbose=False)
        except Exception:          # a handful of scans have inconsistent annotations
            continue
        zvals = scan.slice_zvals
        for n_id, anns in enumerate(nodules):
            i, j, k = np.mean([a.centroid for a in anns], axis=0)
            malignancy = [a.malignancy for a in anns]
            texture = [a.texture for a in anns]
            rows.append({
                'patient_id': scan.patient_id, 'scan_id': scan.id, 'nodule_id': n_id,
                'study_uid': scan.study_instance_uid, 'series_uid': scan.series_instance_uid,
                'slice_thickness': scan.slice_thickness,
                'n_readers': len(anns),
                'malignancy_ratings': ','.join(map(str, malignancy)),
                'malignancy_mean': float(np.mean(malignancy)),
                'malignancy_sd': float(np.std(malignancy)),
                'malignancy_class': malignancy_to_class(np.mean(malignancy)),
                'texture_ratings': ','.join(map(str, texture)),
                'texture_mean': float(np.mean(texture)),
                'texture_class': U.texture_to_class(np.mean(texture)),
                'diameter_mm': float(np.mean([a.diameter for a in anns])),
                'center_row': i, 'center_col': j,
                'center_z': float(np.interp(k, np.arange(len(zvals)), zvals)),   # physical z (mm)
            })
    return pd.DataFrame(rows)


def reader_pairs(nodules: pd.DataFrame, column: str = 'malignancy_ratings') -> np.ndarray:
    """All pairs of ratings given by different radiologists to the same nodule."""
    pairs = []
    for ratings in nodules[column]:
        r = [int(x) for x in str(ratings).split(',')]
        pairs += [(r[a], r[b]) for a in range(len(r)) for b in range(a + 1, len(r))]
    return np.array(pairs)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Images and features
# ─────────────────────────────────────────────────────────────────────────────
def load_series(folder: Path):
    """Read one DICOM series as a 3D volume (z, y, x) in HU, plus the SimpleITK image."""
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(folder)))
    image = reader.Execute()
    return sitk.GetArrayFromImage(image), image


def process_scan(scan_nodules: pd.DataFrame, dims: int = 3) -> list[dict]:
    """Features for every nodule of one LIDC scan — same steps as LNDb's `process_ct`.

    Besides the features, the automatic segmentation is compared (Dice) with the radiologists'
    consensus outline (voxels drawn by at least half of the readers), as Notebook 2 did with the
    LNDb masks.

    `dims=2` computes the 2D features instead (largest axial slice of the same segmentation) — used
    only by Notebook 7 to compare 2D and 3D.
    """
    first = scan_nodules.iloc[0]
    volume, image = load_series(series_dir(first.patient_id, first.study_uid, first.series_uid))
    sx, sy, sz = image.GetSpacing()
    sp = (sz, sy, sx)
    origin = image.GetOrigin()
    scan = pl.query(pl.Scan).filter(pl.Scan.id == int(first.scan_id)).first()
    clusters = scan.cluster_annotations(verbose=False)
    same_slices = len(scan.slice_zvals) == volume.shape[0]

    rows = []
    for _, n in scan_nodules.iterrows():
        base = {'patient_id': n.patient_id, 'scan_id': int(n.scan_id), 'nodule_id': int(n.nodule_id)}
        try:
            iz = image.TransformPhysicalPointToIndex((origin[0], origin[1], float(n.center_z)))[2]
            center = (iz, int(round(n.center_row)), int(round(n.center_col)))
            patch, c = F.crop_patch(volume, center, sp)
            auto, status = F.segment_nodule(patch, c, sp)
            if dims == 2:
                rows.append({**base, 'seg_status': status, **F.features_2d(patch, auto, c, sp)})
                continue
            row = {**base, 'seg_status': status,
                   **F.composition_features(patch, auto, c, sp), **F.radiomic_features(patch, auto, sp)}

            # Dice with the radiologists' consensus (only when slice lists agree, to be safe)
            row['dice_vs_radiologist'] = np.nan
            if same_slices:
                mask_ijk, bbox, _ = consensus(clusters[int(n.nodule_id)], clevel=0.5)
                full = np.zeros(volume.shape, dtype=bool)
                full[bbox[2], bbox[0], bbox[1]] = np.transpose(mask_ijk, (2, 0, 1))
                rmask, _ = F.crop_patch(full, center, sp)
                row['dice_vs_radiologist'] = F.dice(auto, rmask)
            rows.append(row)
        except Exception as e:     # one bad nodule must not stop the scan
            rows.append({**base, 'seg_status': 'error', 'error': str(e)[:200]})
    return rows


def extract_features(nodules: pd.DataFrame, n_jobs: int = -1, dims: int = 3) -> pd.DataFrame:
    """Features for all nodules, one scan per job, cached per scan (re-runs skip finished scans).

    `dims=2` gives the 2D features of Notebook 7, cached in a separate folder.
    """
    cache = U.OUTPUT_DIR / 'cache' / (f'lidc_{F.FEATURE_SET}' + ('_2d' if dims == 2 else ''))
    cache.mkdir(parents=True, exist_ok=True)
    path = lambda scan_id: cache / f'scan-{scan_id:04d}.csv'
    todo = [s for s in sorted(nodules.scan_id.unique()) if not path(s).exists()]

    def job(scan_id):
        try:
            rows = process_scan(nodules[nodules.scan_id == scan_id], dims=dims)
        except Exception as e:     # unreadable scan: report, do not cache
            return scan_id, str(e).splitlines()[-1][:200]
        pd.DataFrame(rows).to_csv(path(scan_id), index=False)
        return scan_id, None

    failed = []
    if todo:
        results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(job)(s) for s in todo)
        failed = [(s, err) for s, err in results if err]
        for s, err in failed:
            print(f'⚠️  scan {s} could not be processed: {err}')

    files = [path(s) for s in sorted(nodules.scan_id.unique()) if path(s).exists()]
    out = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    out.attrs['failed_scans'] = [s for s, _ in failed]
    return out
