
# Patch numpy.RankWarning for compatibility with AnniesLasso
import sys
try:
	import numpy
	from numpy.polynomial import RankWarning as PolyRankWarning
	numpy.RankWarning = PolyRankWarning
except Exception:
	pass

import os
import numpy as np
from astropy.io import fits
import AnniesLasso as tc
import matplotlib.pyplot as plt
from AnniesLasso.thecannon.vectorizer.polynomial import PolynomialVectorizer
from AnniesLasso.thecannon.model import CannonModel

class CannonTrainer:
	def __init__(self, folder):
		self.folder = folder.rstrip('/')
		self._load_data()

	def _load_data(self):
		base = os.path.join(self.folder)
		# Find the prefix (e.g., sfh_2000_10_20250826_144749)
		# Assume the folder contains files with the prefix matching the folder name after last '/'
		prefix = os.path.basename(self.folder)
		self.training_set = fits.getdata(os.path.join(base, f"{prefix}_weights.fits"))
		self.flux = np.load(os.path.join(base, f"{prefix}_spectra.npy"))
		self.ivar = np.load(os.path.join(base, f"{prefix}_invvar.npy"))
		self.wav = np.load(os.path.join(base, f"{prefix}_wavelength.npy"))

	@staticmethod
	def softmax(x, axis=1):
		e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
		return e_x / np.sum(e_x, axis=axis, keepdims=True)

	@staticmethod
	def check_data_for_nans_infs(*arrays):
		for i, arr in enumerate(arrays):
			if np.isnan(arr).any():
				print(f"Warning: NaNs found in array {i}")
			if np.isinf(arr).any():
				print(f"Warning: Infs found in array {i}")

	def split_train_test(self):
		q = np.random.randint(0, 10, len(self.training_set)) % 10
		test_set = (q == 1)
		train_set = (q != 1)
		return train_set, test_set

	def train_and_validate(self):
		self.check_data_for_nans_infs(self.training_set, self.flux, self.ivar)
		train_set, test_set = self.split_train_test()
		vectorizer = PolynomialVectorizer(terms=("1", "2", "3", "4", "5", "6", "7", "8", "9", "10"))
		model = CannonModel(self.training_set[train_set], self.flux[train_set], self.ivar[train_set],
							vectorizer=vectorizer, dispersion=self.wav)
		model.train()
		validation_set_labels = model.test(self.flux[test_set], self.ivar[test_set])

		# Ensure validation_set_labels is a numpy array
		if isinstance(validation_set_labels, tuple):
			validation_set_labels = validation_set_labels[0]
		if hasattr(validation_set_labels, 'dtype') and getattr(validation_set_labels.dtype, 'fields', None) is not None:
			validation_set_labels = validation_set_labels.view(np.float64).reshape(len(validation_set_labels), -1)
		# If you trained on 9 parameters, append a column of zeros for the 10th
		if validation_set_labels.shape[1] == 9:
			logits = np.hstack([validation_set_labels, np.zeros((validation_set_labels.shape[0], 1))])
		else:
			logits = validation_set_labels
		labels_simplex = self.softmax(logits, axis=1)

		# True labels
		if self.training_set[test_set].shape[1] == 9:
			true_logits = np.hstack([self.training_set[test_set], np.zeros((self.training_set[test_set].shape[0], 1))])
		else:
			true_logits = self.training_set[test_set]
		true_labels_simplex = self.softmax(true_logits, axis=1)

		label_names = [f"label_{i+1}" for i in range(labels_simplex.shape[1])]
		results = {}
		for i, label_name in enumerate(label_names):
			x = true_labels_simplex[:, i]
			y = labels_simplex[:, i]
			abs_diff = np.abs(y - x)
			results[label_name] = np.mean(abs_diff)
			print(f"{label_name}: {np.mean(abs_diff):.5f}")
		return results, labels_simplex, true_labels_simplex

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="Train The Cannon model on SFH data.")
	parser.add_argument("folder", type=str, help="Folder containing the required files.")
	args = parser.parse_args()
	trainer = CannonTrainer(args.folder)
	results, labels_simplex, true_labels_simplex = trainer.train_and_validate()
	# Save results to a .txt file in the input folder
	output_path = os.path.join(args.folder.rstrip('/'), "cannon_summary_statistics.txt")
	with open(output_path, "w") as f:
		f.write("# Mean absolute difference between predicted and true labels (simplex space)\n")
		for label, value in results.items():
			f.write(f"{label}: {value:.5f}\n")
	print(f"Summary statistics saved to {output_path}")
