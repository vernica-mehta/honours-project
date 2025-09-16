#!/usr/bin/env python3
# Class file for training The Cannon on a set of galaxy spectra.
# Can be used for any set of labels for which a training library exists

# imports
import os
import numpy as np
from astropy.io import fits
from AnniesLasso.thecannon.vectorizer.polynomial import PolynomialVectorizer
from AnniesLasso.thecannon.model import CannonModel

# define global variable for current directory
cwd = os.getcwd()

from sklearn.model_selection import KFold


class CannonTrainer:

	def __init__(self, filepath, restrict=False):

		""" Initialise CannonTrainer class.
		
		Parameters
		----------
		filepath : str
			Base filename for input/output
		labels : list
			List of label names for training
		"""

		self.filepath = filepath
		self.restrict = restrict
		self.labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

		if self.restrict:

			data = fits.getdata(f"{cwd}/OUTPUTS/{self.filepath}/{self.filepath}_weights.fits")
			if data.dtype.names is not None:
				# Structured array: take log10 of each column
				self.training_set = np.vstack([np.log10(data[ln]) for ln in self.labels]).T
			else:
				# Regular ndarray
				self.training_set = np.log10(data)
		else:
			self.training_set = fits.getdata(f"{cwd}/OUTPUTS/{self.filepath}/{self.filepath}_weights.fits")

		self.normalised_flux = np.load(f"{cwd}/OUTPUTS/{self.filepath}/{self.filepath}_spectra.npy")
		self.normalised_ivar = np.load(f"{cwd}/OUTPUTS/{self.filepath}/{self.filepath}_invvar.npy")
		self.wavelengths = np.load(f"{cwd}/OUTPUTS/{self.filepath}/{self.filepath}_wavelength.npy")

		self.vectorizer = PolynomialVectorizer(self.labels, 2)

		# Prepare label array for all data
		if isinstance(self.training_set, np.ndarray) and self.training_set.dtype.names is not None:
			self.labels_array_all = np.vstack([self.training_set[ln] for ln in self.labels]).T
		else:
			self.labels_array_all = np.asarray(self.training_set)
			if self.labels_array_all.ndim == 1:
				self.labels_array_all = self.labels_array_all.reshape(-1, len(self.labels))

		# Default: single train/test split as before
		q = np.random.randint(0, len(self.labels), len(self.training_set)) % 10
		self.test_set = (q == 1)
		self.train_set = (q != 1)
		self.labels_array = self.labels_array_all[self.train_set]

		return None


	def _train_and_test_split(self, train_idx, test_idx, prefix=None):
		"""Helper to train and test model on given indices, save results."""
		train_labels = self.labels_array_all[train_idx]
		train_flux = self.normalised_flux[train_idx]
		train_ivar = self.normalised_ivar[train_idx]
		test_flux = self.normalised_flux[test_idx]
		test_ivar = self.normalised_ivar[test_idx]

		model = CannonModel(train_labels, train_flux, train_ivar, 
					   vectorizer=self.vectorizer, dispersion=self.wavelengths)
		
		model.train()
		pred_labels, *_ = model.test(test_flux, test_ivar)
		# True labels for test set
		if isinstance(self.training_set, np.ndarray) and self.training_set.dtype.names is not None:
			true_labels = np.vstack([self.training_set[ln][test_idx] for ln in self.labels]).T
		else:
			true_labels = self.labels_array_all[test_idx]

		# Save
		suffix = "_restricted" if self.restrict else ""
		if prefix is None:
			prefix = f"{cwd}/OUTPUTS/{self.filepath}/{self.filepath}{suffix}"
		else:
			prefix = f"{prefix}{suffix}" if not prefix.endswith(suffix) and self.restrict else prefix
		np.save(f"{prefix}_pred.npy", pred_labels)
		np.save(f"{prefix}_true.npy", true_labels)
		return (pred_labels, true_labels)

	def train_and_test(self):
		train_idx = np.where(self.train_set)[0]
		test_idx = np.where(self.test_set)[0]
		self._train_and_test_split(train_idx, test_idx)
		return

	def cross_validate(self, k, random_seed=42):
		"""Run k-fold cross-validation using sklearn KFold. Save fold files in a tar.gz archive."""
		import tempfile
		import shutil
		import tarfile
		kf = KFold(n_splits=k, shuffle=True, random_state=random_seed)
		all_pred = []
		all_true = []
		suffix = "_restricted" if self.restrict else ""
		# Create a temporary directory for fold files
		with tempfile.TemporaryDirectory() as tmpdir:
			fold_file_paths = []
			for i, (train_idx, test_idx) in enumerate(kf.split(self.labels_array_all)):
				fold_prefix = os.path.join(tmpdir, f"{self.filepath}_cv_fold{i+1}{suffix}")
				pred_labels, true_labels = self._train_and_test_split(
					train_idx, test_idx,
					prefix=fold_prefix)
				all_pred.append(pred_labels)
				all_true.append(true_labels)
				# Record file paths for archiving
				fold_pred_path = f"{fold_prefix}_pred.npy"
				fold_true_path = f"{fold_prefix}_true.npy"
				fold_file_paths.extend([fold_pred_path, fold_true_path])
				print(f"Fold {i+1}/{k}: saved predictions and true labels.")

			if self.restrict:
				all_pred = [10**pred for pred in all_pred]
				all_true = [10**true for true in all_true]

			all_pred = [pred / pred.sum(axis=1)[:, np.newaxis] for pred in all_pred]
			all_pred = np.vstack(all_pred)
			all_true = np.vstack(all_true)

			# Save all_pred and all_true directly to output folder
			out_pred = f"{cwd}/OUTPUTS/{self.filepath}/{self.filepath}_cv_all_pred{suffix}.npy"
			out_true = f"{cwd}/OUTPUTS/{self.filepath}/{self.filepath}_cv_all_true{suffix}.npy"
			np.save(out_pred, all_pred)
			np.save(out_true, all_true)

			# Archive all fold files into a tar.gz in the output folder
			tar_path = f"{cwd}/OUTPUTS/{self.filepath}/{self.filepath}_cv_folds{suffix}.tar.gz"
			with tarfile.open(tar_path, "w:gz") as tar:
				for file_path in fold_file_paths:
					arcname = os.path.basename(file_path)
					tar.add(file_path, arcname=arcname)
			print(f"All fold files archived to {tar_path}.")

		print(f"K-fold cross-validation complete. All predictions and true labels saved.")
		return

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="Train The Cannon on galaxy spectra.")
	parser.add_argument("filepath", type=str, help="Base filename for input/output (without OUTPUTS/ prefix and extension)")
	parser.add_argument("--kfold", type=int, default=0, help="Number of folds for k-fold cross-validation (0 to disable)")
	parser.add_argument("--restrict", action='store_true', help="Whether to use restricted model (flag, default False)")
	args = parser.parse_args()
	trainer = CannonTrainer(args.filepath, restrict=args.restrict)
	if args.kfold and args.kfold > 1:
		trainer.cross_validate(k=args.kfold)
	else:
		trainer.train_and_test()