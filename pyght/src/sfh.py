# Class file for generating galaxy spectra based on star formation history (SFH) input only
# Extract galaxy spectrum at particular epoch, isolated or cumulative
# Assumes galaxy evolution from t=0 to present-day universe age

# ensure fsps is in current directory and set environment variable <SPS_HOME>
import os
if os.path.isdir('fsps'):
    os.environ['SPS_HOME'] = 'fsps'
else:
    raise FileNotFoundError("FSPS directory not found!")

# imports
import fsps
import numpy as np
from astropy.io import fits
import datetime

def gen_rand_sfh(length):

    """ Uses a Dirichlet distribution to generate a random SFH array of desired length.

    Parameters
    ----------
    length : `int`
        Desired length of SFH array of fractions.
    
    Returns
    -------
    arr : `array`
        Random SFH array.
    """

    alpha_value = np.random.uniform(0.1, 5.0)
    alpha = np.full(length, alpha_value)

    arr = np.random.dirichlet(alpha)

    return arr

class SFH():

    def __init__(self, sfh_weights, imf=1, flatten=False):

        """ Initialise SFH class.

        Parameters
        ----------
        sfh_weights : `array-like`
            Fractional weights for each spectrum in galaxy evolution time. Number of bins will be calculated as len(sfh_weights)+1. Sum of fractional weights must equal 1.
        imf : `int`, optional
            Initial mass function (default is 1 i.e. Chabrier IMF).
        """

        self.sfh_weights = sfh_weights
        wsum = np.sum(sfh_weights)
        if not np.isclose(wsum, 1.0):
            raise ValueError("Sum of fractional weights must equal 1.")
        
        self.imf = imf
        if imf not in [0,1,2,3,4,5]:
            raise ValueError("Invalid IMF value. Must be one of [0,1,2,3,4,5].")

        self.flatten = flatten

        sp = fsps.StellarPopulation(
            sfh = 0, # single stellar population
            imf_type = self.imf, # IMF as assigned to Class
            nebemlineinspec = False # turn off nebular emission in spectrum
        )
        self.wav, self.spec = sp.get_spectrum()
        self.wav = self.wav[470:2692] # truncating dataset to 370-570 nm wavelength range
        self.spec = self.spec[:,470:2692]

        bin_arr = np.r_[np.array([0.1, 20, 50, 100, 200, 500])*1e6, np.logspace(9.5, 10.15, 4)]
        self.bins = np.log10(bin_arr)

        ages = sp.ssp_ages
        self.all_spec = {}

        spec_ages = dict(zip(ages, self.spec))
        for key, value in spec_ages.items():
            for t in range(len(self.bins)):
                if key <= self.bins[t] and key > self.bins[t - 1]:
                    self.all_spec.setdefault(self.bins[t], []).append(value)

        pass

    def get_averages(self):

        """ Compute the average spectrum for each time-evolution bin by averaging individual spectra within bounds. 
        
        Returns
        -------
        avg_spec : `dict`
            Dictionary of average spectra for each time-evolution bin.
        """

        avg_spec = dict(zip(self.bins[1:], np.zeros(len(self.bins))))

        for key, value in self.all_spec.items():
            avg_spec[key] = np.vstack(value).mean(axis=0)

        return avg_spec

    def get_spectrum(self, age, weighted=False, cumulative=False):

        """ Compute the average spectrum for the time-evolution bin in which the specified age falls. 

        Parameters
        ----------
        age : `float`
            Age of the galaxy in Gyr.
        weighted : `bool`, optional
            If True, spectrum will be scaled by its respective sfh_weight (default is False).
        cumulative : `bool`, optional
            If True, spectrum will be the cumulative sum of all previous bins i.e. galaxy spectrum at specified age (default is False). `weighted` must also be True.

        Returns
        -------
        wav : `array`
            Wavelengths of the spectrum.
        s : `array`
            Average spectrum for the specified time-evolution bin.
        """

        if cumulative == True and weighted == False:
            raise ValueError("Cumulative spectrum requires weighted=True.")

        age = np.log10(age * 1e9) # converting age in Gyr to log year scale

        for i in range(len(self.bins)):
            if age <= self.bins[i] and age > self.bins[i-1]:
                idx = i - 1

        matrix = self.get_averages()
        vals = list(matrix.values())

        if cumulative:
            for v, w in zip(vals[:idx+1], self.sfh_weights[:idx+1]):
                v *= w
                s = np.vstack(v).sum(axis=1)
        else:
            if weighted:
                vals[idx] *= self.sfh_weights[idx]
                s = vals[idx]
            else:
                s = vals[idx]

        return self.wav, s
    
    def final_spectrum(self):

        """ Get the final spectrum of the galaxy at present-day universe age.

        Returns
        -------
        wav : `array`
            Wavelengths of the spectrum.
        s : `array`
            Final spectrum of the galaxy at present-day universe age.
        """

        averages = self.get_averages()

        for key, w in zip(averages.keys(), self.sfh_weights):
            averages[key] *= w

        s = np.vstack(list(averages.values())).sum(axis=0)

        def moving_average(x, w):
            """Returns the moving average of the input array."""
            return np.convolve(x, np.ones(w), 'same') / w

        if self.flatten: # flatten spectrum for The Cannon
            window = 100
            s_flat = s / moving_average(s, window)
            s = s_flat * moving_average(np.ones_like(s), window) # dealing with edges

            # get inverse variance array from flattened spectrum
            uncertainty = np.std(s) * np.ones_like(s)
            variance = uncertainty**2
            inv_var = 1.0 / variance

            return self.wav, s, inv_var

        else:
            return self.wav, s, None



import multiprocessing
from tqdm import tqdm

def _generate_galaxy(args):
    """Helper function to generate a single galaxy's data and return it."""
    _, sfh_len = args
    weights = gen_rand_sfh(sfh_len)
    galaxy = SFH(weights)
    wav, s, inv_var = galaxy.final_spectrum()
    return (weights, s, inv_var, wav)

def churn_galaxies(n, sfh_len, n_jobs=1):
    """Parallel function to churn galaxies using multiprocessing, saving all results in single files. Shows progress bar."""
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sfh_{n}_{sfh_len}_{now}"
    base = f"OUTPUTS/{filename}"
    os.makedirs(base, exist_ok=True)

    import sys
    args_list = [(i, sfh_len) for i in range(n)]
    results = []
    isatty = sys.stdout.isatty()
    def minimal_progress(i):
        # Print every 10% or last, only if not interactive
        if not isatty and ((i+1) % max(1, n//10) == 0 or (i+1) == n):
            print(f"{i+1}/{n} galaxies generated", flush=True)

    if n_jobs == 1:
        if isatty:
            for args in tqdm(args_list, desc="Galaxies", unit="galaxy"):
                results.append(_generate_galaxy(args))
        else:
            for i, args in enumerate(args_list):
                results.append(_generate_galaxy(args))
                minimal_progress(i)
    else:
        if isatty:
            with multiprocessing.Pool(processes=n_jobs) as pool:
                for result in tqdm(pool.imap(_generate_galaxy, args_list), total=n, desc="Galaxies", unit="galaxy"):
                    results.append(result)
        else:
            with multiprocessing.Pool(processes=n_jobs) as pool:
                for i, result in enumerate(pool.imap(_generate_galaxy, args_list)):
                    results.append(result)
                    minimal_progress(i)

    # Unpack results
    weights_list, s_list, inv_var_list, wav_list = zip(*results)
    weights_arr = np.array(weights_list)
    s_arr = np.array(s_list)
    inv_var_arr = np.array(inv_var_list) if inv_var_list is not None else None
    wav_out = np.array(wav_list[0])  # All wav should be identical

    weights_filename = os.path.join(base, f"{filename}_weights.fits")
    hdu = fits.PrimaryHDU(weights_arr)
    hdu.writeto(weights_filename, overwrite=True)

    s_filename = os.path.join(base, f"{filename}_spectra.npy")
    invvar_filename = os.path.join(base, f"{filename}_invvar.npy") if inv_var_arr is not None else None
    wav_filename = os.path.join(base, f"{filename}_wavelength.npy")
    np.save(s_filename, s_arr)
    if invvar_filename is not None:
        np.save(invvar_filename, inv_var_arr)
    np.save(wav_filename, wav_out)
    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Churn given number of galaxies, using SFH of uniform given length.")
    parser.add_argument("n", type=int, help="Number of galaxies to generate.")
    parser.add_argument("sfh_len", type=int, help="Length of uniform SFH array.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel processes to use (default: 1)")
    args = parser.parse_args()

    churn_galaxies(args.n, args.sfh_len, n_jobs=args.n_jobs)