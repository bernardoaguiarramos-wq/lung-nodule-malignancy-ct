"""
download_lidc.py — Download only the LIDC-IDRI scans this project needs (~20 GB instead of ~130 GB).

Lab IA & CD – Project 1 | Lung nodule malignancy on CT (LNDb + LIDC-IDRI)

Why a selection?
    The full LIDC-IDRI collection is ~130 GB. Its annotations (each radiologist's 1–5 malignancy
    rating, plus texture and the nodule outlines) ship with the `pylidc` library, so we chose the
    scans BEFORE downloading anything. `lidc_selection.csv` lists the CT series in priority order:
    most annotated nodules per GB first, with malignant ratings (4–5, the rarest) counting double.
    The `cumulative_gb` column is an ESTIMATE of the space needed (slices × 0.5 MB).

What it does
    Downloads each series from The Cancer Imaging Archive (TCIA) into

        LIDC-IDRI/<patient_id>/<study_uid>/<series_uid>/*.dcm

    which is the folder layout `pylidc` looks for first. Series already downloaded are skipped,
    so the script can be stopped and restarted at any time.

Usage (inside the `labiacd` environment, from the project folder):
    pip install tcia_utils pylidc
    python download_lidc.py --first 3        # quick test with 3 series
    python download_lidc.py                  # default budget: ~20 GB (~248 patients, ~1350 nodules)
    python download_lidc.py --max-gb 12      # smaller budget if disk space is short (up to 30)

LIDC-IDRI licence: CC BY 3.0 — cite Armato et al. (2011), Medical Physics 38(2):915–931,
and the TCIA collection (doi:10.7937/K9/TCIA.2015.LO9QL9SX).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from tcia_utils import nbia

ROOT = Path(__file__).resolve().parent
SELECTION = ROOT / 'lidc_selection.csv'
DEST = ROOT / 'LIDC-IDRI'


def already_downloaded(folder: Path) -> bool:
    return folder.exists() and any(folder.glob('*.dcm'))


def main(max_gb: float, first: int | None = None) -> None:
    series = pd.read_csv(SELECTION)
    series = series[series.cumulative_gb <= max_gb]
    if first:
        series = series.head(first)
    print(f'Estimated size: ~{series.est_gb.sum():.1f} GB')
    print(f'{len(series)} series from {series.patient_id.nunique()} patients → {DEST}')

    failed = []
    for i, row in enumerate(series.itertuples(), start=1):
        study_dir = DEST / row.patient_id / row.study_uid
        if already_downloaded(study_dir / row.series_uid):
            print(f'[{i:3d}/{len(series)}] {row.patient_id}  already downloaded')
            continue
        start = time.time()
        try:
            # tcia_utils writes the files into <path>/<series_uid>/
            nbia.downloadSeries([row.series_uid], input_type='list', path=str(study_dir))
            n = len(list((study_dir / row.series_uid).glob('*.dcm')))
            if n == 0:                                 # tcia_utils logs errors instead of raising
                raise RuntimeError('no DICOM files received (network or server error)')
            print(f'[{i:3d}/{len(series)}] {row.patient_id}  {n} slices  ({time.time() - start:.0f} s)')
        except Exception as error:                      # keep going; report at the end
            failed.append(row.patient_id)
            print(f'[{i:3d}/{len(series)}] {row.patient_id}  FAILED: {error}')

    print('\nDone.' if not failed else f'\nFinished with {len(failed)} failures: {failed}\n'
          'Run the script again to retry them.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--max-gb', type=float, default=20, help='download budget in GB (default 20, max 30)')
    parser.add_argument('--first', type=int, default=None, help='download only the first N series (for testing)')
    args = parser.parse_args()
    main(args.max_gb, args.first)
