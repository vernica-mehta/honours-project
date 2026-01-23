# Class file for generating galaxy spectra based on star formation history (SFH) input only
# Assumes galaxy evolution from t=0 to present-day universe age

# ensure fsps is in current directory and set environment variable <SPS_HOME>
import os
if os.path.isdir('fsps'):
    os.environ['SPS_HOME'] = 'fsps'
elif os.path.isdir('pyght'):
    os.environ['SPS_HOME'] = 'pyght/src/fsps'
else:
    raise FileNotFoundError("FSPS directory not found!")

# imports
import numpy as np
from astropy.io import fits
import datetime

def _gen_rand_sfh(length=10):
    """Generate random star formation history weights."""
    alpha_value = np.random.uniform(0.1, 5.0)
    alpha = np.full(length, alpha_value)
    weights = np.random.dirichlet(alpha)
    return weights

def _moving_average(x, w):
    """Returns the moving average of the input array."""
    return np.convolve(x, np.ones(w), 'same') / w

class SFH():

    def __init__(self, sfh_weights):

        self.data = np.load('/home/vmehta/honours-project/pyght/data/sfh_binned_spectra.npy')
        self.sfh_weights = sfh_weights
        self.wav = np.load('/home/vmehta/honours-project/pyght/data/wavelengths.npy')

        return
    
    def final_spectrum(self, flatten=False):

        s = self.sfh_weights @ self.data

        if flatten:
            # Continuum normalisation for The Cannon
            window = 100
            s_flat = s / _moving_average(s, window)
            s = s_flat * _moving_average(np.ones_like(s), window) # dealing with edges

        return self.wav, s

import multiprocessing
from tqdm import tqdm

def _generate_galaxy():
    """Helper function to generate a single galaxy's data and return it."""
    labels = _gen_rand_sfh(10)
    galaxy = SFH(labels)
    w, s = galaxy.final_spectrum()
    return (labels, s, w)

def churn_galaxies(n, n_jobs=1):
    """Parallel function to churn galaxies using multiprocessing, saving all results in single files. Shows progress bar."""
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sfh_{n}_{now}"
    base = f"/avatar/vmehta/{filename}"
    os.makedirs(base, exist_ok=True)

    import sys
    results = []
    isatty = sys.stdout.isatty()
    def minimal_progress(i):
        # Print every 10% or last, only if not interactive
        if not isatty and ((i+1) % max(1, n//10) == 0 or (i+1) == n):
            print(f"{i+1}/{n} galaxies generated", flush=True)

    if n_jobs == 1:
        if isatty:
            for i in tqdm(range(n), desc="Galaxies", unit="galaxy"):
                results.append(_generate_galaxy())
        else:
            for i in range(n):
                results.append(_generate_galaxy())
                minimal_progress(i)
    else:
        if isatty:
            with multiprocessing.Pool(processes=n_jobs) as pool:
                for result in tqdm(pool.starmap(_generate_galaxy, [() for _ in range(n)]), total=n, desc="Galaxies", unit="galaxy"):
                    results.append(result)
        else:
            with multiprocessing.Pool(processes=n_jobs) as pool:
                for i, result in enumerate(pool.starmap(_generate_galaxy, [() for _ in range(n)])):
                    results.append(result)
                    minimal_progress(i)

    # Unpack results
    labels_list, s_list, wav_list = zip(*results)
    labels_arr = np.array(labels_list)
    s_arr = np.array(s_list)
    wav_out = np.array(wav_list[0])  # All wav should be identical

    labels_filename = os.path.join(base, f"{filename}_labels.fits")
    hdu = fits.PrimaryHDU(labels_arr)
    hdu.writeto(labels_filename, overwrite=True)

    s_filename = os.path.join(base, f"{filename}_spectra.npy")
    wav_filename = os.path.join(base, f"{filename}_wavelength.npy")
    np.save(s_filename, s_arr)
    np.save(wav_filename, wav_out)
    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Churn given number of galaxies, using SFH of uniform given length.")
    parser.add_argument("n", type=int, help="Number of galaxies to generate.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel processes to use (default: 1)")
    args = parser.parse_args()

    churn_galaxies(args.n, n_jobs=args.n_jobs)