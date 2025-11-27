import os
import numpy as np
from astropy.io import fits
from pyght.src.AnniesLasso.thecannon.vectorizer.polynomial import PolynomialVectorizer
from pyght.src.AnniesLasso.thecannon.model import CannonModel
import matplotlib.pyplot as plt
import pandas as pd

class UniformCannonTrainer:

    def __init__(self, filepath, size):

        self.filepath = filepath
        self.size = size
        self.num_train = 500 / self.size

        data = fits.getdata(f"/data/mustard/vmehta/{self.filepath}/{self.filepath}_weights.fits")
        if data.dtype.names is not None:
            # Structured array: take log10 of each column
            self.training_set = np.vstack([np.log10(data[ln]) for ln in self.labels]).T
        else:
            # Regular ndarray
            self.training_set = np.log10(data)
               
        self.all_spectra = np.load(f"/data/mustard/vmehta/{self.filepath}/{self.filepath}_snr_spectra.npy")
        self.all_invvar = np.load(f"/data/mustard/vmehta/{self.filepath}/{self.filepath}_snr_invvar.npy")
        self.wavelengths = np.load(f"/data/mustard/vmehta/{self.filepath}/{self.filepath}_wavelength.npy")
        self.labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        self.vectorizer = PolynomialVectorizer(self.labels, 2)

        if isinstance(self.training_set, np.ndarray) and self.training_set.dtype.names is not None:
            self.labels_array_all = np.vstack([self.training_set[ln] for ln in self.labels]).T
        else:
            self.labels_array_all = np.asarray(self.training_set)
            if self.labels_array_all.ndim == 1:
                self.labels_array_all = self.labels_array_all.reshape(-1, len(self.labels))

        return None
    
    def get_test_set(self):

        if not os.path.exists(f"/data/mustard/vmehta/{self.filepath}/train_test_set/"):

            os.makedirs(f"/data/mustard/vmehta/{self.filepath}/train_test_set/")

            np.random.seed(42)
            test_idx = np.random.default_rng().choice(1000, size=500, replace=False)
            test_idx =np.sort(test_idx)

            test_weights = self.labels_array_all[test_idx]
            test_spectra = self.all_spectra[test_idx]
            test_invvar = self.all_invvar[test_idx]

            train_weights = np.delete(self.labels_array_all, test_idx, axis=0)
            train_spectra = np.delete(self.all_spectra, test_idx, axis=0)
            train_invvar = np.delete(self.all_invvar, test_idx, axis=0)

            np.save(f"/data/mustard/vmehta/{self.filepath}/train_test_set/test_weights.npy", test_weights)
            np.save(f"/data/mustard/vmehta/{self.filepath}/train_test_set/test_spectra.npy", test_spectra)
            np.save(f"/data/mustard/vmehta/{self.filepath}/train_test_set/test_invvar.npy", test_invvar)
            np.save(f"/data/mustard/vmehta/{self.filepath}/train_test_set/train_weights.npy", train_weights)
            np.save(f"/data/mustard/vmehta/{self.filepath}/train_test_set/train_spectra.npy", train_spectra)
            np.save(f"/data/mustard/vmehta/{self.filepath}/train_test_set/train_invvar.npy", train_invvar)

        self.test_weights = np.load(f"/data/mustard/vmehta/{self.filepath}/train_test_set/test_weights.npy")
        self.test_spectra = np.load(f"/data/mustard/vmehta/{self.filepath}/train_test_set/test_spectra.npy")
        self.test_invvar = np.load(f"/data/mustard/vmehta/{self.filepath}/train_test_set/test_invvar.npy")
        self.train_weights = np.load(f"/data/mustard/vmehta/{self.filepath}/train_test_set/train_weights.npy")
        self.train_spectra = np.load(f"/data/mustard/vmehta/{self.filepath}/train_test_set/train_spectra.npy")
        self.train_invvar = np.load(f"/data/mustard/vmehta/{self.filepath}/train_test_set/train_invvar.npy")
        return None

    def train_and_test(self):

        self.get_test_set()
        pred_labels_all = []

        for i in range(int(self.num_train)):

            train_idx = np.random.default_rng().choice(500, size=self.size, replace=False)
            train_idx = np.sort(train_idx)

            train_labels = self.train_weights[train_idx]
            train_flux = self.train_spectra[train_idx]
            train_ivar = self.train_invvar[train_idx]

            model = CannonModel(train_labels, train_flux, train_ivar,
                                vectorizer=self.vectorizer, dispersion=self.wavelengths)
            model.train()

            pred, *_ = model.test(self.test_spectra, self.test_invvar, prior_sum_target=1, prior_sum_std=0.1)
            pred_labels_all.append(pred)

        pred_labels_all = np.array(pred_labels_all)
        pred_labels = np.mean(pred_labels_all, axis=0)
        true_labels = self.test_weights

        return pred_labels, true_labels
    
    def save_results(self):

        pred_labels, true_labels = self.train_and_test()
        np.save(f"/data/mustard/vmehta/{self.filepath}/{self.size}_pred_labels.npy", pred_labels)
        np.save(f"/data/mustard/vmehta/{self.filepath}/true_labels.npy", true_labels) if not os.path.exists(f"/data/mustard/vmehta/{self.filepath}/true_labels.npy") else None
        
        return None

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Train and test The Cannon model with uniform sampling.")
    parser.add_argument("filepath", type=str, help="Path to the dataset.")
    parser.add_argument("size", type=int, help="Training set size.")
    args = parser.parse_args()

    trainer = UniformCannonTrainer(args.filepath, args.size)
    trainer.save_results()

class UniformCannonTester:

    def __init__(self, filepath, size):

        self.filepath = filepath
        self.size = size

        self.real_labels_all = np.load(f"/data/mustard/vmehta/{self.filepath}/true_labels.npy")
        self.pred_labels_all = np.load(f"/data/mustard/vmehta/{self.filepath}/{self.size}_pred_labels.npy")

        bin_arr = np.r_[np.array([0.1, 20, 50, 100, 200, 500])*1e6, np.logspace(9.5, 10.15, 5)]
        binning = np.log10(bin_arr)
        self.bin_widths = np.diff(binning)
        self.bin_centers = binning[:-1] + self.bin_widths/2

        return None
    
    def rmse(self):

        from sklearn.metrics import mean_squared_error

        rmse = np.sqrt(mean_squared_error(self.real_labels_all, self.pred_labels_all, multioutput='raw_values'))
        overall_rmse = np.sqrt(mean_squared_error(self.real_labels_all, self.pred_labels_all))

        df = pd.DataFrame({
            "Label": range(1,11),
            "RMSE": rmse,
        })
        df.loc[len(df.index)] = ["Overall", overall_rmse]

        return df
    
    def plot_difference(self):

        diff = self.pred_labels_all - self.real_labels_all
        median = np.median(diff, axis=0)
        sixteen = np.percentile(diff, 16, axis=0)
        eightyfour = np.percentile(diff, 84, axis=0)

        plt.figure(figsize=(10,5))
        plt.errorbar(self.bin_centers, median, yerr=[median - sixteen, eightyfour - median], fmt='o', capsize=5)
        plt.xlabel('log(Age)')
        plt.ylabel('Difference (pred - true)')
        plt.xlim(2.5,10.5)
        plt.title(f"SFH Difference Plot (Training Size = {self.size})")
        plt.show()
        return None
    
    def t_test(self):

        from scipy.stats import ttest_rel

        pvalues = []
        for i in range(10):
            pred = self.pred_labels_all[:,i]
            real = self.real_labels_all[:,i]

            t, p = ttest_rel(pred, real)
            pvalues.append(p)

        df = pd.DataFrame({
            "Label": range(1,11),
            "P-values": pvalues,
        })

        return df
    
    def bootstrap(self, metric=np.mean, nbr_runs=1000):

        import bstrap

        pvalues = []

        for i in range(10):
            real = pd.DataFrame(self.real_labels_all[:,i])
            pred = pd.DataFrame(self.pred_labels_all[:,i])

            m1, m2, p = bstrap.bootstrap(metric, real, pred, nbr_runs=nbr_runs)
            pvalues.append(p)

        df = pd.DataFrame({
            'Label': range(1, 11),
            'P-values': pvalues,
        })

        return df

class spec_stats(UniformCannonTester):

    def __init__(self, filepath, size, n):

        from pyght.src.sfh import SFH

        super().__init__(filepath, size)
        self.n = n

        self.pred_labels = self.pred_labels_all[self.n]
        self.real_labels = self.real_labels_all[self.n]
        self.real_spec = np.load(f"/data/mustard/vmehta/{self.filepath}/train_test_set/test_spectra.npy")[self.n]
        self.sfh_pred = SFH(self.pred_labels)
        w, self.pred_spec, *_ = self.sfh_pred.final_spectrum()
        self.wavelengths = np.load(f"/data/mustard/vmehta/{self.filepath}/{self.filepath}_wavelength.npy")

        return None
    
    def plot_spectra(self):

        fig, ax = plt.subplots(3,1,figsize=(20,7), constrained_layout=True)

        ax[0].plot(self.wavelengths, self.real_spec, 'r', alpha=0.5)
        ax[0].plot(self.wavelengths, self.pred_spec, 'k', linewidth=0.5)
        ax[0].set_xlim(3700,4200)

        ax[1].plot(self.wavelengths, self.real_spec, 'r', alpha=0.5)
        ax[1].plot(self.wavelengths, self.pred_spec, 'k', linewidth=0.5)
        ax[1].set_xlim(4200,4700)

        ax[2].plot(self.wavelengths, self.real_spec, 'r', alpha=0.5)
        ax[2].plot(self.wavelengths, self.pred_spec, 'k', linewidth=0.5)
        ax[2].set_xlim(4700,5200)
        ax[2].legend(["True Spectrum", "Predicted Spectrum"], loc='lower right')

        fig.suptitle(f"Spectra Comparison - Index {self.n} (Training Size = {self.size})")
        fig.supxlabel("Wavelength (Angstroms)")
        fig.supylabel("Normalized Flux")

        return None

    def plot_sfh(self):

        plt.figure(figsize=(10, 5))
        plt.bar(self.bin_centers, self.real_labels, width=self.bin_widths, align='center', color='gray', alpha=0.5, edgecolor='k', label='f_real')
        plt.bar(self.bin_centers, self.pred_labels, width=self.bin_widths, align='center', color='b', alpha=0.3, edgecolor='b', label='f_pred')
        plt.xlabel('log(Age)')
        plt.ylabel('Weight')
        plt.xlim(2.5,10.5)
        plt.legend()
        plt.title(f"SFH Comparison - Index {self.n} (Training Size = {self.size})")
        plt.show()

        return None 
    

