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
import time

def gen_rand_sfh():

    """ Uses a Dirichlet distribution to generate a random SFH array with length from 5 to 10.

    Returns
    -------
    arr : `array`
        Random SFH array.
    """
    print("Generating random SFH...")

    length = np.random.randint(5, 11)

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
            imf_type = self.imf
        )
        self.wav, self.spec = sp.get_spectrum()

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



if __name__ == "__main__":
    sfh = SFH(None)

def churn_galaxies(t):

    """ Function to churn galaxies, generating random SFHs and computing their spectra.
    Input time in hours.
    Generated weighted and spectra will be outputted to a .fits file.
    """

    t_end = time.time() + 3600 * t

    weights_list = []
    s_list = []

    while time.time() < t_end:

        weights = gen_rand_sfh()
        galaxy = SFH(weights)
        wav, s = galaxy.final_spectrum()

        weights_list.append(weights)
        s_list.append(s)

    col1 = fits.Column(name='SFH function', format='PD()', array=weights_list)
    col2 = fits.Column(name='Spectrum', format='PD()', array=s_list)
    col3 = fits.Column(name='Wavelengths', format='PD()', array=[wav])

    hdu = fits.BinTableHDU.from_columns([col1, col2, col3])
    hdu.writeto('sfh_spectra.fits', overwrite=True)

    return
