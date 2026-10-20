"""
lndb_features.py — Automatic nodule segmentation and feature extraction.

Lab IA & CD – Project 1 | Lung nodule malignancy on CT (LNDb + LIDC-IDRI)

The competition test set only gives the CENTROID of each finding (no segmentation).
To make training and test data comparable, every nodule — train and test — goes
through exactly the same steps:

    1. Crop a 48 mm box around the centroid.
    2. Segment the nodule automatically (adaptive threshold + lung mask + leak control).
    3. Extract radiomic features (pyradiomics) inside that segmentation.
    4. Add a few clinically motivated "composition" features (how much of the
       nodule is solid vs ground-glass, in Hounsfield Units).

For training nodules we ALSO extract the same features inside the radiologist's mask
and measure the overlap (Dice) between both segmentations. This quantifies how much
is lost when no radiologist is available to draw the nodule.

The heavy lifting lives in this module (not in the notebook) so that CT scans can be
processed in parallel and the notebook stays focused on explaining the results.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from joblib import Parallel, delayed
from scipy import ndimage as ndi

import lndb_utils as U

# ─────────────────────────────────────────────────────────────────────────────
# Parameters (chosen by measuring Dice against radiologist masks — see Notebook 2)
# ─────────────────────────────────────────────────────────────────────────────
PATCH_HALF_MM = 24.0   # box of 48 mm around the centroid
SEARCH_RADIUS = 20.0   # the nodule must lie within 20 mm of the centroid
SEED_RADIUS   = 3.0    # the segmented region must touch a 3 mm sphere at the centroid
ALPHA         = 0.3    # threshold = lung + ALPHA · (nodule core − lung)
LUNG_CLOSE_MM = 10.0   # closing radius used to build the lung mask
OPEN_RADII    = (0, 0.75, 1.5, 2.25, 3.0)   # progressively cut thin attachments (vessels)
FALLBACK_MM   = 3.0    # sphere used when no region can be found

# Image versions on which pyradiomics computes features
#   'original' → 107 features (original image only)
#   'full'     → 1130 features: original (107) + 8 wavelet decompositions (8 × 93)
#                + Laplacian of Gaussian at σ = 1, 2, 3 mm (3 × 93)
FEATURE_SET = 'full'
LOG_SIGMAS  = [1.0, 2.0, 3.0]   # mm — fine, medium and coarse structures

# Hounsfield Unit ranges used for the composition features
HU_SOLID = -300        # above: solid tissue
HU_GGO   = -750        # between GGO and solid: ground-glass

logging.getLogger('radiomics').setLevel(logging.ERROR)


# ─────────────────────────────────────────────────────────────────────────────
# Morphology in millimetres (Euclidean distance transforms are exact and fast)
# ─────────────────────────────────────────────────────────────────────────────
def _dilate(m, r, sp):
    return ndi.distance_transform_edt(~m, sampling=sp) <= r


def _erode(m, r, sp):
    return ndi.distance_transform_edt(m, sampling=sp) > r


def _opening(m, r, sp):
    return _dilate(_erode(m, r, sp), r, sp) & m


def _closing(m, r, sp):
    pad = int(np.ceil(r / min(sp))) + 1
    p = np.pad(m, pad, mode='edge')
    out = _erode(_dilate(p, r, sp), r, sp)
    return out[pad:-pad, pad:-pad, pad:-pad]


def _distance_map(shape, center, sp):
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return np.sqrt(((z - center[0]) * sp[0]) ** 2 + ((y - center[1]) * sp[1]) ** 2
                   + ((x - center[2]) * sp[2]) ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Patch extraction
# ─────────────────────────────────────────────────────────────────────────────
def crop_patch(volume: np.ndarray, center: tuple, spacing_zyx: tuple) -> tuple[np.ndarray, tuple]:
    """Crop a box of ±PATCH_HALF_MM around `center` (z, y, x indices), clipped at the borders.

    Returns the patch and the index of the centre inside the patch.
    """
    half = [int(np.ceil(PATCH_HALF_MM / s)) for s in spacing_zyx]
    lo = [max(0, c - h) for c, h in zip(center, half)]
    hi = [min(n, c + h + 1) for n, c, h in zip(volume.shape, center, half)]
    patch = volume[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    return patch, tuple(c - l for c, l in zip(center, lo))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Automatic segmentation
# ─────────────────────────────────────────────────────────────────────────────
def segment_nodule(patch: np.ndarray, center: tuple, spacing_zyx: tuple,
                   return_steps: bool = False):
    """Segment the nodule at `center` without any manual annotation.

    Steps
    -----
    a. Smooth lightly to reduce noise.
    b. Adaptive threshold halfway (ALPHA) between the nodule core and the surrounding
       lung — a single fixed threshold fails because ground-glass nodules (≈ −700 HU)
       are much fainter than solid ones (≈ 0 HU).
    c. Keep only voxels inside the lung (air region closed with a 10 mm ball). This stops
       nodules touching the chest wall from "leaking" into muscle and ribs.
    d. Keep the connected region touching the centroid. If it reaches the edge of the
       search sphere it is probably attached to a vessel: cut thin connections with a
       progressively larger morphological opening until it no longer leaks.
    e. If nothing is found, fall back to a 3 mm sphere.

    Returns (mask, status) — status is 'open<r>', 'clipped' or 'fallback'.
    With return_steps=True also returns intermediate images for visualisation.
    """
    sp = spacing_zyx
    smooth = ndi.gaussian_filter(patch.astype(np.float32), sigma=[0.5 / s for s in sp])
    dist = _distance_map(patch.shape, center, sp)

    core = min(float(np.percentile(smooth[dist <= 2.0], 75)), 100.0)   # cap calcifications
    ring = (dist > 15) & (dist <= 22) & (smooth < -400)
    lung_hu = float(np.median(smooth[ring])) if ring.any() else -850.0
    threshold = lung_hu + ALPHA * (core - lung_hu)

    lung = _closing(smooth < -500, LUNG_CLOSE_MM, sp)
    candidate = (smooth > threshold) & (dist <= SEARCH_RADIUS) & lung

    mask, status, last = None, 'fallback', None
    for r in OPEN_RADII:
        region = _opening(candidate, r, sp) if r > 0 else candidate
        labels, _ = ndi.label(region)
        touching = np.unique(labels[(dist <= SEED_RADIUS) & (labels > 0)])
        if len(touching) == 0:
            break
        best = max(touching, key=lambda l: np.sum((labels == l) & (dist <= SEED_RADIUS)))
        last = labels == best
        if not np.any(last & (dist >= SEARCH_RADIUS - 1.5)):
            mask, status = last, f'open{r}'
            break
    if mask is None and last is not None:
        mask, status = last & (dist <= 10), 'clipped'
    if mask is None:
        mask = dist <= FALLBACK_MM
    mask = ndi.binary_fill_holes(mask)

    if return_steps:
        steps = {'smooth': smooth, 'lung': lung, 'candidate': candidate,
                 'threshold': threshold, 'lung_hu': lung_hu, 'core_hu': core}
        return mask, status, steps
    return mask, status


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Overlap between two binary masks: 1 = identical, 0 = no overlap."""
    return float(2 * np.sum(a & b) / (a.sum() + b.sum())) if (a.sum() + b.sum()) else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# 3. Features
# ─────────────────────────────────────────────────────────────────────────────
_EXTRACTOR = None


def get_extractor():
    """pyradiomics extractor, created once per process.

    - Resampled to 1×1×1 mm so features do not depend on each scanner's voxel size.
    - Bin width 25 HU, the usual choice for CT radiomics.
    - Image versions (FEATURE_SET = 'full'):
        * Original  — the CT itself.
        * Wavelet   — 8 combinations of low (L) / high (H) frequency filters along x, y, z.
                      High-pass versions highlight edges and fine texture; low-pass the overall shape.
        * LoG       — Laplacian of Gaussian at σ = 1, 2, 3 mm: enhances blobs/edges of
                      increasing size (fine → coarse texture).
      Shape features are computed once (they depend only on the mask); every other family
      (93 features) is computed on each of the 12 image versions → 107 + 11 × 93 = 1130.
    """
    global _EXTRACTOR
    if _EXTRACTOR is None:
        import radiomics
        from radiomics import featureextractor
        radiomics.setVerbosity(logging.ERROR)          # silence informational warnings
        logging.getLogger('radiomics').setLevel(logging.ERROR)
        ex = featureextractor.RadiomicsFeatureExtractor(
            binWidth=25, resampledPixelSpacing=[1, 1, 1], interpolator='sitkBSpline', label=1)
        ex.disableAllImageTypes()
        ex.enableImageTypeByName('Original')
        if FEATURE_SET == 'full':
            ex.enableImageTypeByName('Wavelet')
            ex.enableImageTypeByName('LoG', customArgs={'sigma': LOG_SIGMAS})
        ex.disableAllFeatures()
        for cls in ['shape', 'firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']:
            ex.enableFeatureClassByName(cls)
        _EXTRACTOR = ex
    return _EXTRACTOR


def composition_features(patch: np.ndarray, mask: np.ndarray, center: tuple, spacing_zyx: tuple) -> dict:
    """Clinically motivated features: radiologists call a nodule part-solid when it mixes a
    solid component (> −300 HU) with ground-glass (−750 to −300 HU)."""
    hu = patch[mask].astype(np.float32)
    core = patch[_distance_map(patch.shape, center, spacing_zyx) <= 2.0].astype(np.float32)
    voxel_mm3 = float(np.prod(spacing_zyx))
    return {
        'comp_solid_fraction'  : float(np.mean(hu > HU_SOLID)),
        'comp_ggo_fraction'    : float(np.mean((hu > HU_GGO) & (hu <= HU_SOLID))),
        'comp_solid_volume_log': float(np.log1p(np.sum(hu > HU_SOLID) * voxel_mm3)),
        'comp_core_hu_median'  : float(np.median(core)),
        'comp_hu_p90'          : float(np.percentile(hu, 90)),
        'comp_volume_log'      : float(np.log1p(mask.sum() * voxel_mm3)),
    }


_EXTRACTOR_2D = None


def get_extractor_2d():
    """The 2D counterpart of `get_extractor`, used only for the 2D-vs-3D comparison (Notebook 7).

    Same settings (bin width 25 HU, 1 mm pixels, Original + Wavelet + LoG σ = 1, 2, 3 mm), applied to a
    single axial slice: the filters work in-plane (4 wavelet combinations instead of 8) and the shape
    features describe the 2D outline (`shape2D`).
    """
    global _EXTRACTOR_2D
    if _EXTRACTOR_2D is None:
        import radiomics
        from radiomics import featureextractor
        radiomics.setVerbosity(logging.ERROR)
        logging.getLogger('radiomics').setLevel(logging.ERROR)
        ex = featureextractor.RadiomicsFeatureExtractor(
            binWidth=25, resampledPixelSpacing=[1, 1], interpolator='sitkBSpline', label=1)
        ex.disableAllImageTypes()
        ex.enableImageTypeByName('Original')
        ex.enableImageTypeByName('Wavelet')
        ex.enableImageTypeByName('LoG', customArgs={'sigma': LOG_SIGMAS})
        ex.disableAllFeatures()
        for cls in ['shape2D', 'firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']:
            ex.enableFeatureClassByName(cls)
        _EXTRACTOR_2D = ex
    return _EXTRACTOR_2D


def features_2d(patch: np.ndarray, mask: np.ndarray, center: tuple, spacing_zyx: tuple) -> dict:
    """2D features from the axial slice where the segmented nodule is largest.

    This is how a 2D analysis is usually done (a radiologist measures the nodule on its largest
    cross-section). The SAME 3D segmentation is used, so the only difference with the 3D features
    is that a single slice is described instead of the whole volume.
    """
    areas = mask.reshape(mask.shape[0], -1).sum(axis=1)
    z = int(np.argmax(areas)) if areas.max() > 0 else int(center[0])
    sl_img, sl_mask = patch[z], mask[z]
    if sl_mask.sum() == 0:                       # empty segmentation: fall back to the centre pixel
        sl_mask = np.zeros_like(sl_mask)
        sl_mask[int(center[1]), int(center[2])] = True
    image = sitk.GetImageFromArray(sl_img.astype(np.float32))
    image.SetSpacing((float(spacing_zyx[2]), float(spacing_zyx[1])))
    roi = sitk.GetImageFromArray(sl_mask.astype(np.uint8))
    roi.CopyInformation(image)
    out = get_extractor_2d().execute(image, roi)
    feats = {f'r2d_{k}': float(v) for k, v in out.items() if not k.startswith('diagnostics_')}
    # Composition on the slice: 'volume' becomes the area in mm² (pixel area × 1)
    comp = composition_features(sl_img[None], sl_mask[None], (0, int(center[1]), int(center[2])),
                                (1.0, spacing_zyx[1], spacing_zyx[2]))
    rename = {'comp_volume_log': 'c2d_area_log', 'comp_solid_volume_log': 'c2d_solid_area_log'}
    feats.update({rename.get(k, k.replace('comp_', 'c2d_')): v for k, v in comp.items()})
    return feats


def radiomic_features(patch: np.ndarray, mask: np.ndarray, spacing_zyx: tuple) -> dict:
    image = sitk.GetImageFromArray(patch.astype(np.float32))
    image.SetSpacing(tuple(float(s) for s in spacing_zyx[::-1]))   # SimpleITK uses (x, y, z)
    roi = sitk.GetImageFromArray(mask.astype(np.uint8))
    roi.CopyInformation(image)
    out = get_extractor().execute(image, roi)
    # Keep the numeric features only (drop pyradiomics' 'diagnostics_' metadata).
    # Names: rad_<image version>_<family>_<feature>, e.g. rad_wavelet-HLL_glcm_Contrast
    return {f'rad_{k}': float(v) for k, v in out.items() if not k.startswith('diagnostics_')}


# ─────────────────────────────────────────────────────────────────────────────
# 4. One CT at a time (unit of parallelism and caching)
# ─────────────────────────────────────────────────────────────────────────────
def process_ct(ct_path: Path, findings: pd.DataFrame, with_radiologist: bool) -> list[dict]:
    """Extract features for every finding of one CT.

    findings: rows of that CT with LNDbID, FindingID, x, y, z and — when
              with_radiologist — is_small, mask_rad_id, mask_finding_id.
    """
    ct = sitk.ReadImage(str(ct_path))
    volume = sitk.GetArrayFromImage(ct)
    sx, sy, sz = ct.GetSpacing()
    sp = (sz, sy, sx)
    rad_masks = {}
    rows = []

    for _, f in findings.iterrows():
        base = {'LNDbID': int(f.LNDbID), 'FindingID': int(f.FindingID)}
        try:
            ix, iy, iz = ct.TransformPhysicalPointToIndex((float(f.x), float(f.y), float(f.z)))
            patch, c = crop_patch(volume, (iz, iy, ix), sp)
            auto, status = segment_nodule(patch, c, sp)
            row = {**base, 'roi': 'auto', 'seg_status': status,
                   **composition_features(patch, auto, c, sp), **radiomic_features(patch, auto, sp)}

            if with_radiologist and not bool(f.is_small):
                rid = int(f.mask_rad_id)
                if rid not in rad_masks:
                    rad_masks[rid] = sitk.GetArrayFromImage(sitk.ReadImage(str(U.mask_path(int(f.LNDbID), rid))))
                rmask, _ = crop_patch(rad_masks[rid], (iz, iy, ix), sp)
                rmask = rmask == int(f.mask_finding_id)
                row['dice_vs_radiologist'] = dice(auto, rmask)
                rows.append(row)
                if rmask.sum() > 0:
                    rows.append({**base, 'roi': 'radiologist', 'seg_status': 'manual',
                                 **composition_features(patch, rmask, c, sp),
                                 **radiomic_features(patch, rmask, sp)})
            else:
                rows.append(row)
        except Exception as e:  # one bad finding must not stop the whole extraction
            rows.append({**base, 'roi': 'auto', 'seg_status': 'error', 'error': str(e)[:200]})
    return rows


def extract_features(findings: pd.DataFrame, ct_index: dict, split: str,
                     with_radiologist: bool = False, n_jobs: int = -1) -> pd.DataFrame:
    """Extract features for all findings, one CT per job, with a per-CT cache.

    Results of each CT are saved in outputs/cache/<split>_<FEATURE_SET>/LNDb-XXXX.csv, so if
    the run is interrupted, re-running only processes the CTs that are missing. The feature
    set is part of the folder name, so switching sets never mixes old and new results.
    """
    cache = U.OUTPUT_DIR / 'cache' / f'{split}_{FEATURE_SET}'
    cache.mkdir(parents=True, exist_ok=True)
    todo = [i for i in sorted(findings.LNDbID.unique())
            if i in ct_index and not (cache / f'LNDb-{i:04d}.csv').exists()]
    missing = sorted(set(findings.LNDbID) - set(ct_index))

    def job(i):
        # A CT that cannot be read (e.g. corrupted file) is reported, not cached, and does
        # not stop the other CTs — fix the file and re-run to process it.
        try:
            rows = process_ct(ct_index[i], findings[findings.LNDbID == i], with_radiologist)
        except Exception as e:
            return i, str(e).splitlines()[-1][:200]
        pd.DataFrame(rows).to_csv(cache / f'LNDb-{i:04d}.csv', index=False)
        return i, None

    failed = []
    if todo:
        results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(job)(i) for i in todo)
        failed = [(i, err) for i, err in results if err]
        for i, err in failed:
            print(f'⚠️  LNDb-{i:04d} could not be processed: {err}')

    files = [cache / f'LNDb-{i:04d}.csv' for i in sorted(findings.LNDbID.unique())
             if (cache / f'LNDb-{i:04d}.csv').exists()]
    out = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    out.attrs['missing_cts'] = missing
    out.attrs['failed_cts'] = [i for i, _ in failed]
    return out
