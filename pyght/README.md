# pyght

This package delivers a working program to generate galaxy spectra from SFHs and related workflows for the Honours project.

## Quick start

1) Clone with Git LFS

- Git LFS is required to fetch the FSPS data shipped in this repository.
- If you don't have LFS on your machine, install it (no sudo needed on many systems; see Tips below) and then run:

```bash
git lfs install
git lfs pull
```

2) Create and activate a Python environment

```bash
# Python >=3.9 recommended
python -m venv .venv
source .venv/bin/activate

# Or use conda/mamba:
# conda create -n pyght python=3.11 -y && conda activate pyght
```

3) Install dependencies (pip)

```bash
pip install -r pyght/requirements.txt
```

Alternative (conda):

```bash
conda install -c conda-forge python-fsps numpy scipy astropy tqdm
```

4) FSPS data location (important)

- The code expects an `fsps` directory to be present in the current working directory when you run.
- This repository includes the FSPS data at `pyght/src/fsps/`.
- Therefore, run commands from `pyght/src/` so FSPS is found automatically.

```bash
cd pyght/src
```

## Usage

Generate N galaxies with a uniform SFH length and save outputs (uses multiprocessing and shows progress):

```bash
cd pyght/src
python sfh.py <N> <SFH_LEN> --n_jobs <PROCESSES>

# Example: 100 galaxies, SFH length 10, using 4 processes
python sfh.py 100 10 --n_jobs 4
```

Outputs are written under `pyght/src/OUTPUTS/sfh_<N>_<SFH_LEN>_<TIMESTAMP>/` as:
- `<...>_weights.fits` — SFH weights for each galaxy
- `<...>_spectra.npy` — spectra array (N x wavelength)
- `<...>_invvar.npy` — inverse variance (only when flatten=True; see code)
- `<...>_wavelength.npy` — wavelength grid

Notes:
- The spectra are truncated to 370–570 nm in the code (`sfh.py`).
- Nebular emission is disabled by default.

## Tips for Git LFS on shared/remote systems

No sudo? You can still install LFS:

- Conda/mamba: `conda install -c conda-forge git-lfs`
- Prebuilt binary (Linux): download the git-lfs Linux tarball from GitHub Releases and place `git-lfs` in `$HOME/.local/bin` (add to PATH).
- After installing, always run: `git lfs install` and then `git lfs pull` inside the repo.

## Development notes

This folder contains everything related to delivering a usable program that can resolve the star formation history of a galaxy from its present-day spectrum. The initial objective is to separate training and testing, i.e., being able to load a previously trained model to test spectra that may be unrelated to the training data.

The rationale behind working here separately from the main project is to focus on packaging and workflow, while core model development can continue elsewhere in the repository.