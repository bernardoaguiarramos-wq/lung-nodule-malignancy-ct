# Setup — Python environment

Lab IA & CD – Project 1 | Lung nodule malignancy on CT (LNDb + LIDC-IDRI)

All notebooks run in one isolated environment called **`labiacd`** (Python 3.11).
An isolated environment guarantees that the project always uses the same library
versions, regardless of what else is installed on the computer.

## 1. One-time prerequisites (macOS)

`pyradiomics` contains C code that is compiled during installation, so a compiler is needed:

```bash
xcode-select --install
```

(If it says the tools are already installed, move on.)

## 2. Create the environment

Run from inside the repository folder:

```bash
conda create -n labiacd python=3.11 -y
conda activate labiacd
pip install -r requirements.txt
pip install pyradiomics --no-build-isolation
```

`--no-build-isolation` makes pyradiomics compile against the NumPy already installed
(NumPy < 2). Without it, pip builds it against NumPy 2 and the import fails.

## 3. Check that everything works

```bash
python -c "import numpy, pandas, sklearn, skimage, imblearn, SimpleITK, jinja2, pydicom, radiomics; print('ALL OK', radiomics.__version__)"
```

Expected output: `ALL OK v3.0.1` (or a newer version).

## 4. Use it in VS Code

Open a notebook → **Select Kernel** (top right) → **Python Environments** → `labiacd`.

Every time you open a new terminal, activate the environment first:

```bash
conda activate labiacd
```

## Library roles

| Library | Used for |
|---|---|
| numpy, pandas, scipy | Data handling and statistics |
| SimpleITK | Reading CT scans and masks (`.mhd` / `.raw`) |
| scikit-image | Automatic nodule segmentation around the centroid |
| pyradiomics | Radiomic feature extraction (shape, intensity, texture) |
| scikit-learn | Models (KNN, SVM, MLP), validation, metrics |
| imbalanced-learn | Oversampling of the minority classes |
| matplotlib | Figures |
| jinja2 | Formatted result tables in Notebook 4 |
| pydicom | Reading LIDC-IDRI scans (DICOM format) |
| ipykernel | Running the environment inside VS Code notebooks |
