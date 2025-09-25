import numpy as np
import os

cwd = os.getcwd()

class SNR:

    def __init__(self, spectrum, snr):

        self.spectrum = spectrum
        self.snr = snr

        return None

    def add_noise(self):

        if self.snr == None:
            return self.spectrum
        
        signal_power = np.mean(self.spectrum**2)
        noise_power = signal_power / (10**(self.snr/10))
        noise = np.random.normal(0, np.sqrt(noise_power), self.spectrum.shape)

        return self.spectrum + noise

    def flatten_spectrum(self):

        s = self.add_noise()

        def moving_average(x, w):
            """Returns the moving average of the input array."""
            return np.convolve(x, np.ones(w), 'same') / w

        window = 100
        s_flat = s / moving_average(s, window)
        final_s = s_flat * moving_average(np.ones_like(s), window) # dealing with edges

        # get inverse variance array from flattened spectrum
        uncertainty = np.std(final_s) * np.ones_like(final_s)
        variance = uncertainty**2
        inv_var = 1.0 / variance

        return final_s, inv_var
    
def snr_worker(filepath, snr=None):

    s_list = []
    inv_list = []
    spectra_all = np.load(f"{cwd}/OUTPUTS/{filepath}/{filepath}_spectra.npy")
    
    for s in spectra_all:
        s_instance = SNR(s, snr)
        s_flat, inv_var = s_instance.flatten_spectrum()
        s_list.append(s_flat)
        inv_list.append(inv_var)

    s_array = np.array(s_list)
    inv_array = np.array(inv_list)
    
    return s_array, inv_array


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Add noise to spectra and flatten them.")
    parser.add_argument("filepath", type=str, help="Base filename for input/output")
    parser.add_argument("--snr", type=float, default=None, help="Desired signal-to-noise ratio (None for no noise)")
    args = parser.parse_args()
    
    s_array, inv_array = snr_worker(args.filepath, args.snr)
    n = int(args.snr) if args.snr is not None else ''
    np.save(f"{cwd}/OUTPUTS/{args.filepath}/{args.filepath}_snr{n}_spectra.npy", s_array)
    np.save(f"{cwd}/OUTPUTS/{args.filepath}/{args.filepath}_snr{n}_invvar.npy", inv_array)
    print(f"Saved flattened spectra and inverse variance arrays with SNR={n}.")