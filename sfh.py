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
    print("Random SFH generated:", arr)

    return arr

class SFH():

    def __init__(self, sfh_weights, imf=1):

        """ Initialise SFH class.

        Parameters
        ----------
        sfh_weights : `array-like`
            Fractional weights for each spectrum in galaxy evolution time. Number of bins will be calculated as len(sfh_weights)+1. Sum of fractional weights must equal 1.
        imf : `int`, optional
            Initial mass function (default is 1 i.e. Chabrier IMF).
        """

        print("Checking inputs...")

        self.sfh_weights = sfh_weights
        wsum = np.sum(sfh_weights)
        if not np.isclose(wsum, 1.0):
            raise ValueError("Sum of fractional weights must equal 1.")
        
        self.imf = imf
        if imf not in [0,1,2,3,4,5]:
            raise ValueError("Invalid IMF value. Must be one of [0,1,2,3,4,5].")
        
        self.nbins = len(sfh_weights) + 1

        print("Initialising spectra...")

        sp = fsps.StellarPopulation(
            sfh = 0,
            imf_type = self.imf,
            nebemlineinspec = False # turn off nebular emission in spectrum
        )
        self.wav, self.spec = sp.get_spectrum()
        self.wav = self.wav[330:4664] # truncating dataset to 300-900 nm wavelength range
        self.spec = self.spec[:][330:4664]

        print("Initialising binning...")

        self.bins = np.linspace(5.5, 10.15, self.nbins)
        ages = sp.ssp_ages
        self.all_spec = {}

        spec_ages = dict(zip(ages, self.spec))
        for key, value in spec_ages.items():
            for t in range(len(self.bins)):
                if key <= self.bins[t] and key > self.bins[t - 1]:
                    self.all_spec.setdefault(self.bins[t], []).append(value)

        print("Initialisation complete!")
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

        print("Compiling spectrum...")

        averages = self.get_averages()

        for key, w in zip(averages.keys(), self.sfh_weights):
            averages[key] *= w

        s = np.vstack(list(averages.values())).sum(axis=0)
        print("Final spectrum compiled!")

        return self.wav, s

def churn_galaxies(n, sfh_len):

    """ Function to churn galaxies, generating random SFHs and computing their spectra.
    Input time in hours.
    Generated weighted and spectra will be outputted to a .fits file.
    """

    weights_list = []
    s_list = []
    
    for i in range(0,n):

        weights = gen_rand_sfh(sfh_len)
        galaxy = SFH(weights)
        wav, s = galaxy.final_spectrum()

        weights_list.append(weights)
        s_list.append(s)
    
    weights_arr = np.array(weights_list)
    s_arr       = np.array(s_list)

    col1 = fits.Column(name="SFH function", format=f"{weights_arr.shape[1]}D", array=weights_arr)
    col2 = fits.Column(name="Spectrum", format=f"{s_arr.shape[1]}D", array=s_arr)

    table_hdu = fits.BinTableHDU.from_columns([col1, col2])
    wav_hdu   = fits.ImageHDU(data=wav, name="WAVELENGTHS")
    hdul = fits.HDUList([fits.PrimaryHDU(), table_hdu, wav_hdu])


    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"sfh_{n}_{sfh_len}_{now}.fits"
    hdul.writeto(filename, overwrite=True)

    print(f".fits file generated: {filename}")

    return

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Churn given number of galaxies, using SFH of uniform given length.")
    parser.add_argument("n", type=int, help="Number of galaxies to generate.")
    parser.add_argument("sfh_len", type=int, help="Length of uniform SFH array.")
    args = parser.parse_args()

    print("Starting churn...")
    churn_galaxies(args.n, args.sfh_len)

# to run via terminal, use the command `./run_sfh.sh [n] [sfh_len]