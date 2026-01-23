import numpy as np
import os


class SNR:

    def __init__(self, spectrum, snr):

        self.spectrum = spectrum
        self.snr = snr

        return None

    def add_noise(self):

        if self.snr == None:
            return self.spectrum, np.full_like(self.spectrum, 1e-15)
        
        noise = np.mean(self.spectrum) / self.snr
        spec_noise = np.random.normal(self.spectrum, noise)
        noise_array = np.full_like(self.spectrum, noise)

        return spec_noise, noise_array

    def flatten_spectrum(self):

        s, n = self.add_noise()

        def moving_average(x, w):
            """Returns the moving average of the input array."""
            return np.convolve(x, np.ones(w), 'same') / w

        window = 100
        s_flat = s / moving_average(s, window)
        final_s = s_flat * moving_average(np.ones_like(s), window) # dealing with edges

        # get inverse variance array from flattened spectrum
        n_flat = n / moving_average(s, window)
        final_n = n_flat * moving_average(np.ones_like(s), window) # dealing with edges
        inv_var = 1.0 / (final_n**2)

        return final_s, inv_var
    
def snr_worker(filepath, snr=None):

    s_list = []
    inv_list = []
    spectra_all = np.load(f"/avatar/vmehta/{filepath}/{filepath}_spectra.npy")
    
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
    np.save(f"/avatar/vmehta/{args.filepath}/{args.filepath}_snr{n}_spectra.npy", s_array)
    np.save(f"/avatar/vmehta/{args.filepath}/{args.filepath}_snr{n}_invvar.npy", inv_array)
    print(f"Saved flattened spectra and inverse variance arrays with SNR={n}.")