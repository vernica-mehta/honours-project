#!/usr/bin/env python3
# Class file for training The Cannon on a set of galaxy spectra.
# Can be used for any set of labels for which a training library exists

# imports
import os
import numpy as np
from astropy.io import fits
from code.src.AnniesLasso.thecannon.vectorizer.polynomial import PolynomialVectorizer
from code.src.AnniesLasso.thecannon.model import CannonModel

class CannonTrainer:

	def __init__(self, filepath, snr=None, nlabels=None):

		""" Initialise CannonTrainer class.
		
		Parameters
		----------
		filepath : str
			Base filename for input/output
		labels : list
			List of label names for training
		"""

		self.filepath = filepath
		self.labels = list(map(str, range(1,nlabels+1)))
		self.snr = snr

		self.training_set = fits.getdata(f"/avatar/vmehta/binning-tests/{self.filepath}/{self.filepath}_labels.fits")
		self.flux = np.load(f"/avatar/vmehta/binning-tests/{self.filepath}/{self.filepath}_snr{int(self.snr)}_spectra.npy")
		self.ivar = np.load(f"/avatar/vmehta/binning-tests/{self.filepath}/{self.filepath}_snr{int(self.snr)}_invvar.npy")
		self.wavelengths = np.load(f"/avatar/vmehta/binning-tests/{self.filepath}/{self.filepath}_wavelength.npy")
		self.vectorizer = PolynomialVectorizer(self.labels, 2)
		self.labels_array_all = np.asarray(self.training_set)

		return None

	def _iter_manual_kfold(self, k):
		"""Yield (i, train_idx, test_idx) using the same manual scheme as kfold_train."""
		n = len(self.labels_array_all)
		block_size = max(1, n // k)
		for i in range(k):
			start = i * block_size
			end = (i + 1) * block_size if i < (k - 1) else n
			if start >= n:
				break
			train_idx = np.arange(start, min(end, n))
			test_idx = np.concatenate([np.arange(0, start), np.arange(min(end, n), n)])
			yield i, train_idx, test_idx

	def _train_and_test_split(self, train_idx, test_idx, prefix=None, save_model_file=True):
		"""Helper to train and test model on given indices, save results."""

		train_flux = self.flux[train_idx]
		train_ivar = self.ivar[train_idx]
		
		train_labels = self.labels_array_all[train_idx]
		test_flux = self.flux[test_idx]
		test_ivar = self.ivar[test_idx]

		model = CannonModel(train_labels, train_flux, train_ivar, 
					   vectorizer=self.vectorizer, dispersion=self.wavelengths)
		
		model.train()
		
		# Set bounds: each label must be between 0 and 1
		n_labels = len(self.labels)
		bounds = ([0] * n_labels, [1] * n_labels)
		
		pred_labels, *_ = model.test(test_flux, test_ivar, 
									  prior_sum_target=1, prior_sum_std=0.1,
									  label_bounds=bounds)
		true_labels = self.labels_array_all[test_idx]

		# Save predictions/true labels
		if prefix != None:
			prefix = prefix
		else:
			prefix = f"/avatar/vmehta/binning-tests/{self.filepath}/snr{int(self.snr)}"
		np.save(f"{prefix}_pred.npy", pred_labels)
		np.save(f"{prefix}_true.npy", true_labels)
		
		# Optionally save the trained model (per-fold). Default is True for compatibility.
		if save_model_file:
			model_path = f"{prefix}_model.pkl"
			model.write(model_path, overwrite=True)
		
		return (pred_labels, true_labels, model)

	def kfold_train(self):
		"""Manual k-fold using contiguous blocks for training and the rest for testing."""
		for i, train_idx, test_idx in self._iter_manual_kfold(10):
			start, end = train_idx[0], train_idx[-1] + 1
			print(f"Fold {i+1}: train [{start}:{end}), test {len(test_idx)}")
			self._train_and_test_split(train_idx, test_idx)

	def train_and_test(self, random_seed=42):
		"""Run a randomized 20/80 train/test split."""
		n = len(self.labels_array_all)
		split = n // 5
		rng = np.random.default_rng(random_seed)
		perm = rng.permutation(n)
		train_idx = perm[:split]
		test_idx = perm[split:]
		print(f"Train/test split (seed={random_seed}): train {len(train_idx)}, test {len(test_idx)}")
		self._train_and_test_split(train_idx, test_idx)
		return

	def cross_validate(self, k):
		"""Run k-fold cross-validation using the same manual scheme as kfold_train."""
		import tempfile
		import tarfile
		all_pred = []
		all_true = []
		all_models = []
		fold_indices = []
		# Create a temporary directory for fold files
		with tempfile.TemporaryDirectory() as tmpdir:
			fold_file_paths = []
			for i, (train_idx, test_idx) in ((i, (tr, te)) for i, tr, te in self._iter_manual_kfold(k)):
				fold_prefix = os.path.join(tmpdir, f"{self.filepath}_snr{int(self.snr)}_fold{i+1}")
				pred_labels, true_labels, model = self._train_and_test_split(
					train_idx, test_idx,
					prefix=fold_prefix,
					save_model_file=False)
				all_pred.append(pred_labels)
				all_true.append(true_labels)
				all_models.append(model)
				fold_indices.append({"fold": i+1, "train_idx": train_idx.tolist(), "test_idx": test_idx.tolist()})
				# Record file paths for archiving
				fold_pred_path = f"{fold_prefix}_pred.npy"
				fold_true_path = f"{fold_prefix}_true.npy"
				fold_file_paths.extend([fold_pred_path, fold_true_path])
				print(f"Fold {i+1}/{k}: saved predictions and true labels.")

			# Save all_pred and all_true directly to output folder
			out_pred = f"/avatar/vmehta/binning-tests/{self.filepath}/snr{int(self.snr)}_all_pred.npy"
			out_true = f"/avatar/vmehta/binning-tests/{self.filepath}/snr{int(self.snr)}_all_true.npy"
			np.save(out_pred, np.concatenate(all_pred, axis=0))
			np.save(out_true, np.concatenate(all_true, axis=0))

			# Archive all fold files into a tar.gz in the output folder (pred/true only)
			tar_path = f"/avatar/vmehta/binning-tests/{self.filepath}/snr{int(self.snr)}_folds.tar.gz"
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
	parser.add_argument("filepath", type=str, help="Base filename for input/output")
	parser.add_argument("--kfold", type=int, default=None, help="Number of folds for k-fold cross-validation")
	parser.add_argument("--snr", type=float, default=None, help="Signal-to-noise ratio used for noisy spectra")
	parser.add_argument("--nlabels", type=int, default=None, help="Number of labels (bins) used in the SFH")
	args = parser.parse_args()
	trainer = CannonTrainer(args.filepath, snr=args.snr, nlabels=args.nlabels)
	if args.kfold is not None and args.kfold > 1:
		trainer.cross_validate(k=args.kfold)
	else:
		trainer.train_and_test()