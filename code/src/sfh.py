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

def _gen_rand_sfh(iteration, nbins, rng, mix=(0.4, 0.2, 0.4),
                  alpha=(1.0, 0.25), fixed_levels=None):
    """Generate SFH weights on the simplex with mixed coverage modes.

    Modes:
    - uniform: Dirichlet(alpha_uniform) for broad simplex coverage
    - extreme: Dirichlet(alpha_extreme) to emphasize edge/corner solutions
    - grid: cycle one fixed bin through fixed levels; randomize the remainder
    """
    if nbins < 2:
        raise ValueError("nbins must be >= 2.")

    probs = np.asarray(mix, dtype=float)
    if probs.shape != (3,):
        raise ValueError("mix must contain exactly 3 values: [uniform, extreme, grid].")

    alpha_vals = np.asarray(alpha, dtype=float)
    if alpha_vals.shape != (2,):
        raise ValueError("alpha must contain exactly 2 values: [uniform, extreme].")
    alpha_uniform, alpha_extreme = alpha_vals
    if np.any(probs < 0):
        raise ValueError("Sampling probabilities must be non-negative.")
    if probs.sum() == 0:
        raise ValueError("At least one sampling probability must be > 0.")
    probs = probs / probs.sum()

    if fixed_levels is None:
        fixed_levels = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
    else:
        fixed_levels = np.asarray(fixed_levels, dtype=float)
        if np.any((fixed_levels < 0.0) | (fixed_levels > 1.0)):
            raise ValueError("fixed_levels must be between 0 and 1.")

    mode = rng.choice(["uniform", "extreme", "grid"], p=probs)

    if mode == "uniform":
        return rng.dirichlet(np.full(nbins, alpha_uniform))

    if mode == "extreme":
        return rng.dirichlet(np.full(nbins, alpha_extreme))

    # grid mode: deterministic coverage across bins and fixed fractions
    steps_per_cycle = nbins * len(fixed_levels)
    step = iteration % steps_per_cycle
    fixed_bin = step // len(fixed_levels)
    fixed_value = fixed_levels[step % len(fixed_levels)]
    remaining_mass = 1.0 - fixed_value

    weights = np.zeros(nbins, dtype=float)
    weights[fixed_bin] = fixed_value
    other_bins = np.delete(np.arange(nbins), fixed_bin)
    if remaining_mass > 0:
        weights[other_bins] = rng.dirichlet(np.full(nbins - 1, alpha_uniform)) * remaining_mass
    return weights


def _basis_spectra_path(nbins):
    """Return a sensible basis-spectra file path for the requested bin count."""
    root = "/home/vmehta/honours-project/code/data"
    candidates = [
        f"{root}/sfh_binning/{nbins}_bins.npy",
        f"{root}/sfh_{nbins}bins_spectra.npy",
    ]
    if nbins == 3:
        candidates.insert(0, f"{root}/sfh_threebins_spectra.npy")
    if nbins == 4:
        candidates.insert(0, f"{root}/sfh_fourbins_spectra.npy")
    if nbins == 6:
        candidates.insert(0, f"{root}/sfh_sixbins_spectra.npy")
    if nbins == 8:
        candidates.insert(0, f"{root}/sfh_eightbins_spectra.npy")
    if nbins == 10:
        candidates.insert(0, f"{root}/sfh_tenbins_spectra.npy")

    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No basis spectra file found for nbins={nbins}. Tried: {candidates}")

def _moving_average(x, w):
    """Returns the moving average of the input array."""
    return np.convolve(x, np.ones(w), 'same') / w

class SFH():

    def __init__(self, sfh_weights, nbins=2):
        data_raw = np.load(_basis_spectra_path(nbins))
        # Normalize each basis template to comparable continuum scale before mixing.
        # This keeps label interpolation shape-driven rather than dominated by absolute
        # luminosity differences between SSP age bins.
        template_scale = np.median(data_raw, axis=1, keepdims=True)
        template_scale = np.where(template_scale == 0.0, 1.0, template_scale)
        self.data = data_raw / template_scale
        self.sfh_weights = np.asarray(sfh_weights, dtype=float)
        if self.sfh_weights.shape[0] != nbins:
            raise ValueError(f"Expected {nbins} SFH weights, got {self.sfh_weights.shape[0]}.")
        if self.data.shape[0] != nbins:
            raise ValueError(
                f"Basis spectra has {self.data.shape[0]} bins, expected {nbins}.")
        if not np.isclose(self.sfh_weights.sum(), 1.0, atol=1e-8):
            raise ValueError("SFH weights must sum to 1.0.")
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

def _generate_galaxy(labels, nbins):
    """Generate a single galaxy spectrum from an SFH label vector."""
    galaxy = SFH(labels, nbins=nbins)
    w, s = galaxy.final_spectrum()
    return (labels, s, w)

def churn_galaxies(n, nbins, mix=(0.4, 0.4, 0.2), alpha=(1.0, 0.25)):
    """Generate n galaxies with mixed SFH sampling on the simplex."""

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sfh_{n}_{nbins}bins_{now}"
    base = f"/avatar/vmehta/{filename}"
    os.makedirs(base, exist_ok=True)

    import sys
    results = []
    isatty = sys.stdout.isatty()
    rng = np.random.default_rng(42)

    def minimal_progress(i):
        # Print every 10% or last, only if not interactive
        if not isatty and ((i+1) % max(1, n//10) == 0 or (i+1) == n):
            print(f"{i+1}/{n} galaxies generated", flush=True)

    if isatty:
        for i in tqdm(range(n), desc="Galaxies", unit="galaxy"):
            labels = _gen_rand_sfh(
                i, nbins, rng,
                mix=mix,
                alpha=alpha)
            results.append(_generate_galaxy(labels=labels, nbins=nbins))
    else:
        for i in range(n):
            labels = _gen_rand_sfh(
                i, nbins, rng,
                mix=mix,
                alpha=alpha)
            results.append(_generate_galaxy(labels=labels, nbins=nbins))
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
    parser = argparse.ArgumentParser(description="Churn galaxies using mixed simplex SFH sampling.")
    parser.add_argument("n", type=int, help="Number of galaxies to generate.")
    parser.add_argument("--nbins", type=int, default=2, help="Number of SFH bins to use.")
    parser.add_argument(
        "--mix",
        type=float,
        nargs=3,
        metavar=("UNIFORM", "EXTREME", "GRID"),
        default=[1.0, 0.0, 0.0],
        help="Relative weights for sampling modes as three values, e.g. --mix 0.5 0.3 0.2",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        nargs=2,
        metavar=("UNIFORM", "EXTREME"),
        default=[1.0, 0.25],
        help="Dirichlet alpha values as two numbers, e.g. --alpha 1.0 0.25",
    )
    args = parser.parse_args()

    churn_galaxies(
        args.n,
        args.nbins,
        mix=args.mix,
        alpha=args.alpha)