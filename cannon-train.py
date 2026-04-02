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

	def __init__(self, filepath, snr=None, nlabels=None, log_model=False, log_flux=False):

		""" Initialise CannonTrainer class.
		
		Parameters
		----------
		filepath : str
			Base filename for input/output
		labels : list
			List of label names for training
		log_flux : bool
			Whether to take the logarithm of the flux before training
		"""

		self.filepath = filepath
		self.labels = list(map(str, range(1,nlabels+1)))
		self.snr = snr
		self.log_model = log_model
		self.log_flux = log_flux
		self.output_root = f"/avatar/vmehta/{self.filepath}/finalmodeltest"
		#self.output_subdir = "log-model" if self.log_model else "linear-model"
		self.output_subsubdir = "log-flux" if self.log_flux else "linear-flux"
		self.output_dir = os.path.join(self.output_root, self.output_subsubdir)
		os.makedirs(self.output_dir, exist_ok=True)

		self.training_set = fits.getdata(f"/avatar/vmehta/{self.filepath}/{self.filepath}_labels.fits")
		self.flux = np.load(f"/avatar/vmehta/{self.filepath}/{self.filepath}_snr{int(self.snr)}_spectra.npy")
		self.ivar = np.load(f"/avatar/vmehta/{self.filepath}/{self.filepath}_snr{int(self.snr)}_invvar.npy")
		self.wavelengths = np.load(f"/avatar/vmehta/{self.filepath}/{self.filepath}_wavelength.npy")
		self.vectorizer = PolynomialVectorizer(self.labels, 2)
		self.labels_all = np.asarray(self.training_set)

		return None

	def _iter_manual_kfold(self, k):
		"""Yield (i, train_idx, test_idx) with one contiguous test block per fold."""
		n = len(self.labels_all)
		block_size = max(1, n // k)
		for i in range(k):
			start = i * block_size
			end = (i + 1) * block_size if i < (k - 1) else n
			if start >= n:
				break
			test_idx = np.arange(start, min(end, n))
			train_idx = np.concatenate([np.arange(0, start), np.arange(min(end, n), n)])
			yield i, train_idx, test_idx

	def _transform_flux_space(self, flux, ivar):
		"""Return flux/ivar in requested model space (linear or log10 flux)."""
		if not self.log_flux:
			return flux, ivar

		# log10-space requires strictly positive flux; invalid pixels are masked.
		eps = np.finfo(float).tiny
		safe_flux = np.clip(flux, eps, None)
		log_flux = np.log10(safe_flux)

		ln10 = np.log(10.0)
		ivar_log = ivar * (safe_flux * ln10) ** 2
		invalid = (~np.isfinite(flux)) | (~np.isfinite(ivar)) | (flux <= 0.0) | (ivar <= 0.0)
		ivar_log = np.where(invalid, 0.0, ivar_log)
		log_flux = np.where(invalid, 0.0, log_flux)
		return log_flux, ivar_log

	def _train_and_test_split(self, train_idx, test_idx, prefix=None, save_model_file=True):
		"""Helper to train and test model on given indices, save results."""

		train_flux_linear = self.flux[train_idx]
		train_ivar_linear = self.ivar[train_idx]
		
		train_labels_linear = self.labels_all[train_idx]
		test_flux_linear = self.flux[test_idx]
		test_ivar_linear = self.ivar[test_idx]

		train_flux, train_ivar = self._transform_flux_space(train_flux_linear, train_ivar_linear)
		test_flux, test_ivar = self._transform_flux_space(test_flux_linear, test_ivar_linear)

		if self.log_model:
			eps = np.finfo(float).tiny
			train_labels = np.log10(np.clip(train_labels_linear, eps, None))
		else:
			train_labels = train_labels_linear

		model = CannonModel(train_labels, train_flux, train_ivar, 
					   vectorizer=self.vectorizer, dispersion=self.wavelengths)
		
		model.train()
		
		if self.log_model:
			pred_labels_log, *_ = model.test(
				test_flux,
				test_ivar,
				prior_sum_target=1, prior_sum_std=0.1,
				prior_sum_mode="log")
			pred_labels = np.power(10.0, pred_labels_log)
		else:
			n_labels = len(self.labels)
			bounds = ([0] * n_labels, [1] * n_labels)
			pred_labels, *_ = model.test(
				test_flux,
				test_ivar,
				prior_sum_target=1, prior_sum_std=0.1,
				prior_sum_mode="linear",
				label_bounds=bounds)
		true_labels = self.labels_all[test_idx]

		# Save predictions/true labels
		if prefix != None:
			prefix = prefix
		else:
			prefix = os.path.join(self.output_dir, f"snr{int(self.snr)}")
		np.save(f"{prefix}_pred.npy", pred_labels)
		np.save(f"{prefix}_true.npy", true_labels)
		
		# Optionally save the trained model (per-fold). Default is True for compatibility.
		if save_model_file:
			model_path = f"{prefix}_model.pkl"
			model.write(model_path, include_training_set_spectra=True, overwrite=True)
		
		return (pred_labels, true_labels, model)

	def kfold_train(self):
		"""Manual k-fold using contiguous blocks for testing and the rest for training."""
		for i, train_idx, test_idx in self._iter_manual_kfold(10):
			test_start, test_end = test_idx[0], test_idx[-1] + 1
			print(f"Fold {i+1}: test [{test_start}:{test_end}), train {len(train_idx)}")
			self._train_and_test_split(train_idx, test_idx)

	def train_and_test(self, random_seed=42):
		"""Run a randomized 80/20 train/test split."""
		n = len(self.labels_all)
		split = int(0.8 * n)
		rng = np.random.default_rng(random_seed)
		perm = rng.permutation(n)
		train_idx = perm[:split]
		test_idx = perm[split:]
		print(f"Train/test split (seed={random_seed}): train {len(train_idx)}, test {len(test_idx)}")
		self._train_and_test_split(train_idx, test_idx)
		return

	def cross_validate(self, k):
		"""Run k-fold cross-validation with one contiguous test block per fold."""
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
				fold_file_paths.extend([
					fold_pred_path,
					fold_true_path,
				])
				print(f"Fold {i+1}/{k}: saved predictions and true labels.")

			# Save predictions and true labels directly to output folder
			out_pred_linear = os.path.join(self.output_dir, f"snr{int(self.snr)}_all_pred.npy")
			out_true_linear = os.path.join(self.output_dir, f"snr{int(self.snr)}_all_true.npy")
			np.save(out_pred_linear, np.concatenate(all_pred, axis=0))
			np.save(out_true_linear, np.concatenate(all_true, axis=0))

			# Archive all fold files into a tar.gz in the output folder (pred/true only)
			tar_path = os.path.join(self.output_dir, f"snr{int(self.snr)}_folds.tar.gz")
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
	parser.add_argument("--log-model", action="store_true", help="Train/test in log10 label space but save linear labels")
	parser.add_argument("--log-flux", action="store_true", help="Train/test in log10 flux space")
	args = parser.parse_args()
	trainer = CannonTrainer(
		args.filepath,
		snr=args.snr,
		nlabels=args.nlabels,
		log_model=args.log_model,
		log_flux=args.log_flux)
	if args.kfold is not None and args.kfold > 1:
		trainer.cross_validate(k=args.kfold)
	else:
		trainer.train_and_test()