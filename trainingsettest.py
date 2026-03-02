import os
import numpy as np
from astropy.io import fits
from code.src.AnniesLasso.thecannon.vectorizer.polynomial import PolynomialVectorizer
from code.src.AnniesLasso.thecannon.model import CannonModel

class UniformCannonTrainer:

    def __init__(self, filepath, size):

        self.filepath = filepath
        self.size = size
        self.num_train = 500 / self.size

        data = fits.getdata(f"/avatar/vmehta/{self.filepath}/{self.filepath}_labels.fits")
        if data.dtype.names is not None:
            # Structured array: take log10 of each column
            self.training_set = np.vstack([np.log10(data[ln]) for ln in self.labels]).T
        else:
            # Regular ndarray
            self.training_set = np.log10(data)
               
        self.all_spectra = np.load(f"/avatar/vmehta/{self.filepath}/{self.filepath}_snr_spectra.npy")
        self.all_invvar = np.load(f"/avatar/vmehta/{self.filepath}/{self.filepath}_snr_invvar.npy")
        self.wavelengths = np.load(f"/avatar/vmehta/{self.filepath}/{self.filepath}_wavelength.npy")
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

        if not os.path.exists(f"/avatar/vmehta/{self.filepath}/train_test_set/"):

            os.makedirs(f"/avatar/vmehta/{self.filepath}/train_test_set/")

            np.random.seed(42)
            test_idx = np.random.default_rng().choice(1000, size=500, replace=False)
            test_idx =np.sort(test_idx)

            test_weights = self.labels_array_all[test_idx]
            test_spectra = self.all_spectra[test_idx]
            test_invvar = self.all_invvar[test_idx]

            train_weights = np.delete(self.labels_array_all, test_idx, axis=0)
            train_spectra = np.delete(self.all_spectra, test_idx, axis=0)
            train_invvar = np.delete(self.all_invvar, test_idx, axis=0)

            np.save(f"/avatar/vmehta/{self.filepath}/train_test_set/test_weights.npy", test_weights)
            np.save(f"/avatar/vmehta/{self.filepath}/train_test_set/test_spectra.npy", test_spectra)
            np.save(f"/avatar/vmehta/{self.filepath}/train_test_set/test_invvar.npy", test_invvar)
            np.save(f"/avatar/vmehta/{self.filepath}/train_test_set/train_weights.npy", train_weights)
            np.save(f"/avatar/vmehta/{self.filepath}/train_test_set/train_spectra.npy", train_spectra)
            np.save(f"/avatar/vmehta/{self.filepath}/train_test_set/train_invvar.npy", train_invvar)

        self.test_weights = np.load(f"/avatar/vmehta/{self.filepath}/train_test_set/test_weights.npy")
        self.test_spectra = np.load(f"/avatar/vmehta/{self.filepath}/train_test_set/test_spectra.npy")
        self.test_invvar = np.load(f"/avatar/vmehta/{self.filepath}/train_test_set/test_invvar.npy")
        self.train_weights = np.load(f"/avatar/vmehta/{self.filepath}/train_test_set/train_weights.npy")
        self.train_spectra = np.load(f"/avatar/vmehta/{self.filepath}/train_test_set/train_spectra.npy")
        self.train_invvar = np.load(f"/avatar/vmehta/{self.filepath}/train_test_set/train_invvar.npy")
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
            print(f"Completed train/test iteration {i+1}/{int(self.num_train)}")

        pred_labels_all = np.array(pred_labels_all)
        pred_labels = np.mean(pred_labels_all, axis=0)
        true_labels = self.test_weights

        return pred_labels, true_labels
    
    def save_results(self):

        pred_labels, true_labels = self.train_and_test()
        np.save(f"/avatar/vmehta/{self.filepath}/{self.size}_pred_labels.npy", pred_labels)
        np.save(f"/avatar/vmehta/{self.filepath}/true_labels.npy", true_labels) if not os.path.exists(f"/avatar/vmehta/{self.filepath}/true_labels.npy") else None
        
        return None

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Train and test The Cannon model with uniform sampling.")
    parser.add_argument("filepath", type=str, help="Path to the dataset.")
    parser.add_argument("size", type=int, help="Training set size.")
    args = parser.parse_args()

    trainer = UniformCannonTrainer(args.filepath, args.size)
    trainer.save_results()