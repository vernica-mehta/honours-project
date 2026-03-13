# Class file for generating galaxy spectra based on star formation history (SFH) input only
# Assumes galaxy evolution from t=0 to present-day universe age

# ensure fsps is in current directory and set environment variable <SPS_HOME>
import os
if os.path.isdir('fsps'):
    os.environ['SPS_HOME'] = 'fsps'
elif os.path.isdir('code'):
    os.environ['SPS_HOME'] = 'code/src/fsps'
else:
    raise FileNotFoundError("FSPS directory not found!")

# imports
import numpy as np
from astropy.io import fits
import datetime

def _gen_rand_sfh(iteration, nbins):
    """Generate controlled-random SFH weights for a given iteration.

    Pattern per cycle:
    - For each bin (0 .. nbins-1)
    - Fix that bin at 0.0, 0.1, ..., 1.0
    - Randomise the remaining bins with a Dirichlet draw so total sum stays 1.0
    """
    if nbins < 2:
        raise ValueError("nbins must be >= 2 to randomise remaining bins.")

    levels = np.round(np.arange(0.0, 1.01, 0.1), 1)  # 0.0 .. 1.0
    steps_per_cycle = nbins * len(levels)
    step = iteration % steps_per_cycle

    fixed_bin = step // len(levels)
    fixed_value = levels[step % len(levels)]

    remaining_mass = 1.0 - fixed_value
    other_bins = np.delete(np.arange(nbins), fixed_bin)

    alpha_value = np.random.uniform(0.1, 5.0)
    alpha = np.full(nbins - 1, alpha_value)
    random_other_weights = np.random.dirichlet(alpha) * remaining_mass

    weights = np.zeros(nbins)
    weights[fixed_bin] = fixed_value
    weights[other_bins] = random_other_weights
    return weights

def _moving_average(x, w):
    """Returns the moving average of the input array."""
    return np.convolve(x, np.ones(w), 'same') / w

class SFH():

    def __init__(self, sfh_weights, nbins):

        #self.data = np.load('/home/vmehta/honours-project/code/data/sfh_binned_spectra.npy')
        self.data = np.load(f'/home/vmehta/honours-project/code/data/sfh_binning/{nbins}_bins.npy')
        self.sfh_weights = sfh_weights
        self.wav = np.load('/home/vmehta/honours-project/code/data/wavelengths.npy')

        return
    
    def final_spectrum(self, flatten=False):

        s = self.sfh_weights @ self.data

        if flatten:
            # Continuum normalisation for The Cannon
            window = 100
            s_flat = s / _moving_average(s, window)
            s = s_flat * _moving_average(np.ones_like(s), window) # dealing with edges

        return self.wav, s

from tqdm import tqdm

def _generate_galaxy(seed=None, nbins=None, n_total=None):
    """Helper function to generate a single galaxy's data and return it."""
    # Reseed RNG to ensure unique random state in each process/call
    if seed is not None:
        np.random.seed(seed)

    if nbins == 2:
        if n_total is None or n_total < 1:
            raise ValueError("n_total must be provided and >= 1 when nbins == 2.")
        idx = seed if seed is not None else 0
        first_label = np.linspace(0.0, 1.0, n_total)[idx]
        labels = np.array([first_label, 1.0 - first_label])
    else:
        labels = _gen_rand_sfh(seed if seed is not None else 0, nbins)

    galaxy = SFH(labels, nbins)
    w, s = galaxy.final_spectrum()
    return (labels, s, w)

def churn_galaxies(n, nbins):
    """Parallel function to churn galaxies using multiprocessing, saving all results in single files. Shows progress bar."""
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sfh_{n}_{nbins}bins_{now}"
    base = f"/avatar/vmehta/binning-tests/{filename}"
    os.makedirs(base, exist_ok=True)

    import sys
    results = []
    isatty = sys.stdout.isatty()
    def minimal_progress(i):
        # Print every 10% or last, only if not interactive
        if not isatty and ((i+1) % max(1, n//10) == 0 or (i+1) == n):
            print(f"{i+1}/{n} galaxies generated", flush=True)

    if isatty:
        for i in tqdm(range(n), desc="Galaxies", unit="galaxy"):
            results.append(_generate_galaxy(seed=i, nbins=nbins, n_total=n))
    else:
        for i in range(n):
            results.append(_generate_galaxy(seed=i, nbins=nbins, n_total=n))
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
    parser.add_argument("--nbins", type=int, help="Number of SFH bins to use.")
    args = parser.parse_args()

    churn_galaxies(args.n, args.nbins)