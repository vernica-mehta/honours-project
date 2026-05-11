# The Cannon_SFH

This repository contains tools used to synthesise galaxy spectra from star-formation histories (SFHs), build spectral libraries, add realistic noise, and train Cannon models on those spectra. The base functionality of _The Cannon_ is forked from [here](https://github.com/andycasey/AnniesLasso).

**Quick summary:**
- Generate base spectra using [FSPS](https://github.com/cconroy20/fsps)
- Build spectral libraries from the SFH utilities
- Add noise to spectra using `snr` helpers
- Train/test The Cannon models with `cannon-train.py`

**Prerequisites**

- Python 3.8+
- Install runtime dependencies listed in `code/requirements.txt`:

```bash
python3 -m pip install -r code/requirements.txt
```

**Repository layout**

- `code/` — main project code and tests
	- `code/src/` — Python modules and scripts (e.g. `sfh.py`, `snr.py`, `cannon-train.py`)
	- `code/data/` — tracked data assets (binned spectra, wavelengths, example FITS)
	- `code/notebooks/` — interactive workflows and examples (e.g. `sfh.ipynb`)
	- `code/tests/` — lightweight test scripts / helpers
- `thesis/` submodule - for easier transfer of information between this repo and my thesis document

Files you will commonly use:

- [code/notebooks/sfh.ipynb](code/notebooks/sfh.ipynb) — interactive notebook for generating base spectra from SFHs. Good for exploration and visual checks.
- [code/src/sfh.py](code/src/sfh.py) — functions to synthesise spectral libraries programmatically.
- [code/src/snr.py](code/src/snr.py) — utilities to add noise to spectra (produce SNR variants).
- [code/src/cannon-train.py](code/src/cannon-train.py) — CLI entrypoint to train and evaluate The Cannon models.

Basic Workflows
---------------

1) Generate base spectra (interactive)

- Open the notebook [code/notebooks/sfh.ipynb](code/notebooks/sfh.ipynb) in Jupyter / VS Code notebooks and run the cells to produce example base spectra and diagnostic plots. The notebook walks through SFH definitions, synthesising spectra and saving outputs into `code/data/`.

2) Build a spectral library (scripted)

- Use the library functions in [code/src/sfh.py](code/src/sfh.py) to produce and save spectral libraries programmatically. Example pattern (from CLI):

```shell
python3 code/src/sfh.py 1000 --nbins 6
```

Refer to the top of `code/src/sfh.py` for available helpers and argument descriptions.

3) Add noise (SNR) to spectra

- Use the utilities in [code/src/snr.py](code/src/snr.py) to add noise and create SNR-specific `.npy` files that match the pipeline's expected naming convention (e.g. `..._snr100_spectra.npy`). Example (from Python):

```py
from code.src import snr
snr.add_noise(input_spectra_path, output_spectra_path, snr=100)
```

Check `code/src/snr.py` for exact function names and parameters.

4) Train and evaluate The Cannon

- The main entrypoint is [code/src/cannon-train.py](code/src/cannon-train.py). Example CLI invocations:

```bash
# simple train/test split on dataset basename 'mydataset' with SNR=100 and 6 labels
python3 code/src/cannon-train.py mydataset --snr 100 --nlabels 6

# k-fold cross-validation
python3 code/src/cannon-train.py mydataset --snr 100 --nlabels 6 --kfold 10

# train on one dataset, test on another
python3 code/src/cannon-train.py mydataset --snr 100 --nlabels 6 --train-filepath trainset --test-filepath testset
```

The script saves prediction arrays and trained model files into the output directory under `/avatar/vmehta/<dataset>/finalmodel/` (see script header for exact locations).

Notes & recommendations
-----------------------

- Data tracking: `code/data/` is intentionally tracked in this repository; avoid re-adding large external datasets here.
- Use a virtual environment (pyenv/venv) per the repository's development workflow.
- If you alter the data naming conventions, update `cannon-train.py` and scripts that assume `_snr{N}_spectra.npy` naming.
