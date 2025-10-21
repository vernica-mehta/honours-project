import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import bstrap
plt.style.use('custom.mplstyle')
from pyght.src.sfh import SFH

class CannonData:
    """
    Loads and stores all data that does NOT depend on n.
    """
    def __init__(self, base, restricted=False, use_intersection=True):
        import os
        self.restricted = restricted
        output_dir = f"OUTPUTS/{base}/"
        suffix = "_restricted" if restricted else ""

        self.f_real = np.load(f"{output_dir}{base}_cv_all_true.npy")
        self.f_pred = np.load(f"{output_dir}{base}_cv_all_pred{suffix}.npy")

        # If intersection requested and both pred files exist, use intersection of valid indices
        if use_intersection:
            pred_file_unres = f"{output_dir}{base}_cv_all_pred.npy"
            pred_file_res = f"{output_dir}{base}_cv_all_pred_restricted.npy"
            if os.path.exists(pred_file_unres) and os.path.exists(pred_file_res):
                f_pred_unres = np.load(pred_file_unres)
                f_pred_res = np.load(pred_file_res)
                nan_rows_unres = np.isnan(f_pred_unres).any(axis=1)
                nan_rows_res = np.isnan(f_pred_res).any(axis=1)
                valid_indices = np.where(~nan_rows_unres & ~nan_rows_res)[0]
            else:
                nan_rows = np.isnan(self.f_pred).any(axis=1)
                valid_indices = np.where(~nan_rows)[0]
        else:
            nan_rows = np.isnan(self.f_pred).any(axis=1)
            valid_indices = np.where(~nan_rows)[0]

        self.valid_indices = valid_indices
        self.pred_clean = self.f_pred[valid_indices]
        self.true_clean = self.f_real[valid_indices]

        # Bin setup (used in multiple places)
        bin_arr = np.r_[np.array([0.001, 0.1, 20, 50, 100, 200, 500])*1e6, np.logspace(9.5, 10.15, 4)]
        binning = np.log10(bin_arr)
        self.bin_widths = np.diff(binning)
        self.bin_centers = binning[:-1] + self.bin_widths/2

    def rmse(self):
        rmse = np.sqrt(mean_squared_error(self.true_clean, self.pred_clean, multioutput='raw_values'))
        overall_rmse = np.sqrt(mean_squared_error(self.true_clean, self.pred_clean))
        return rmse, overall_rmse

    @property
    def rmse_n(self):
        rmse_list = []
        for i in range(len(self.true_clean)):
            rmse_n = np.sqrt(mean_squared_error(self.true_clean[i], self.pred_clean[i]))
            rmse_list.append(rmse_n)
        return np.array(rmse_list)

    def plot_difference(self):
        diff = self.pred_clean - self.true_clean
        median = np.median(diff, axis=0)
        sixteen = np.percentile(diff, 16, axis=0)
        eightyfour = np.percentile(diff, 84, axis=0)

        plt.figure(figsize=(10,5))
        plt.errorbar(self.bin_centers, median, yerr=[median - sixteen, eightyfour - median], fmt='o', capsize=5)
        plt.xlabel('log(Age)')
        plt.ylabel('Difference (pred - true)')
        plt.xlim(2.5,10.5)
        plt.title(f"SFH Difference - {'Restricted' if self.restricted else 'Unrestricted'}")
        plt.show()
        return None

class CannonInstance:
    """
    Handles per-n logic, using a CannonData object.
    """
    def __init__(self, data: CannonData, n):
        self.data = data
        self.n = n
        self.global_idx = data.valid_indices[n]
        self.sfh_real = SFH(data.f_real[self.global_idx])
        self.sfh_pred = SFH(data.pred_clean[n])
        self.wav_real, self.s_real, *_ = self.sfh_real.final_spectrum()
        self.wav_pred, self.s_pred, *_ = self.sfh_pred.final_spectrum()

    def plot_spectra(self):
        fig, ax = plt.subplots(3,1,figsize=(20,7))
        ax[0].plot(self.wav_real, self.s_real, 'r', alpha=0.5)
        ax[0].plot(self.wav_pred, self.s_pred, 'k', linewidth=0.5)
        ax[0].set_xlim(3700,4200)
        ax[1].plot(self.wav_real, self.s_real, 'r', alpha=0.5)
        ax[1].plot(self.wav_pred, self.s_pred, 'k', linewidth=0.5)
        ax[1].set_xlim(4200,4700)
        ax[2].plot(self.wav_real, self.s_real, 'r', alpha=0.5)
        ax[2].plot(self.wav_pred, self.s_pred, 'k', linewidth=0.5)
        ax[2].set_xlim(4700,5200)
        fig.suptitle(f"Spectra Comparison - {'Restricted' if self.data.restricted else 'Unrestricted'}")
        return None

    def plot_sfh(self):
        real_weights = self.data.f_real[self.global_idx]
        pred_weights = self.data.pred_clean[self.n]
        plt.figure(figsize=(10, 5))
        plt.bar(self.data.bin_centers, real_weights, width=self.data.bin_widths, align='center', color='b', alpha=0.5, edgecolor='b', label='f_real')
        plt.bar(self.data.bin_centers, pred_weights, width=self.data.bin_widths, align='center', color='r', alpha=0.5, edgecolor='r', label='f_pred')
        plt.xlabel('log(Age)')
        plt.ylabel('Weight')
        plt.xlim(2.5,10.5)
        plt.legend()
        plt.title(f"SFH Comparison - {'Restricted' if self.data.restricted else 'Unrestricted'}")
        plt.show()
        return None

class CompareModels:
    """
    Compares restricted and unrestricted models.
    """
    def __init__(self, base, use_intersection=True):
        self.restricted_data = CannonData(base, restricted=True, use_intersection=use_intersection)
        self.unrestricted_data = CannonData(base, restricted=False, use_intersection=use_intersection)
        self.bin_centers = self.restricted_data.bin_centers
        self.bin_widths = self.restricted_data.bin_widths

    # Functions that do NOT require n
    def compare_difference(self):
        diff_restricted = self.restricted_data.pred_clean - self.restricted_data.true_clean
        diff_unrestricted = self.unrestricted_data.pred_clean - self.unrestricted_data.true_clean
        median_restricted = np.median(diff_restricted, axis=0)
        sixteen_restricted = np.percentile(diff_restricted, 16, axis=0)
        eightyfour_restricted = np.percentile(diff_restricted, 84, axis=0)
        median_unrestricted = np.median(diff_unrestricted, axis=0)
        sixteen_unrestricted = np.percentile(diff_unrestricted, 16, axis=0)
        eightyfour_unrestricted = np.percentile(diff_unrestricted, 84, axis=0)
        plt.figure(figsize=(10,5))
        plt.errorbar(self.bin_centers, median_restricted, yerr=[median_restricted - sixteen_restricted, eightyfour_restricted - median_restricted], fmt='o', capsize=5, color='b', label='Restricted Model')
        plt.errorbar(self.bin_centers, median_unrestricted, yerr=[median_unrestricted - sixteen_unrestricted, eightyfour_unrestricted - median_unrestricted], fmt='o', capsize=5, color='r', label='Unrestricted Model')
        plt.hlines(0, 2.5, 10.5, colors='k', linestyles='dashed', alpha=0.5)
        plt.xlabel('log(Age)')
        plt.ylabel('Difference (pred - true)')
        plt.xlim(2.5,10.5)
        plt.title("SFH Difference Comparison (all plots):")
        plt.legend()
        plt.show()
        return None

    def compare_rmse(self):
        rmse_restricted, overall_restricted = self.restricted_data.rmse()
        rmse_unrestricted, overall_unrestricted = self.unrestricted_data.rmse()
        labels = [f"{j+1}" for j in range(len(rmse_restricted))] + ["Overall"]
        restricted_vals = list(rmse_restricted) + [overall_restricted]
        unrestricted_vals = list(rmse_unrestricted) + [overall_unrestricted]
        df = pd.DataFrame({
            "Label": labels,
            "Restricted": restricted_vals,
            "Unrestricted": unrestricted_vals
        })
        # print("RMSE Comparison (all plots):")
        # print(df.to_string(index=False))
        return df
    
    def t_test(self):

        res_pvalues = []
        unres_pvalues = []

        for i in range(10):
            pred_res = self.restricted_data.pred_clean[:,i]
            pres_unres = self.unrestricted_data.pred_clean[:,i]
            real = self.restricted_data.true_clean[:,i]

            t_res, p_res = ttest_rel(pred_res, real)
            t_unres, p_unres = ttest_rel(pres_unres, real)
            res_pvalues.append(p_res)
            unres_pvalues.append(p_unres)

        df = pd.DataFrame({
            "Label": range(1,11),
            "Restricted": res_pvalues,
            "Unrestricted": unres_pvalues
        })

        return df
    
    def bootstrap(self, metric=np.mean, nbr_runs=1000):
        res_pvals = []
        unres_pvals = []

        for i in range(10):
            real = pd.DataFrame(self.restricted_data.true_clean[:,i])
            pred_res = pd.DataFrame(self.restricted_data.pred_clean[:,i])
            pred_unres = pd.DataFrame(self.unrestricted_data.pred_clean[:,i])

            m1, m2, pval_res = bstrap.bootstrap(metric, real, pred_res, nbr_runs=nbr_runs)
            m1, m2, pval_unres = bstrap.bootstrap(metric, real, pred_unres, nbr_runs=nbr_runs)
            res_pvals.append(pval_res)
            unres_pvals.append(pval_unres)

        df = pd.DataFrame({
            'Label': range(1, 11),
            'Restricted': res_pvals,
            'Unrestricted': unres_pvals
        })

        return df

    @property
    def compare_rmse_n(self):
        restricted_list = []
        unrestricted_list = []
        for i in range(len(self.restricted_data.true_clean)):
            rmse_n_restricted = np.sqrt(mean_squared_error(self.restricted_data.true_clean[i], self.restricted_data.pred_clean[i]))
            restricted_list.append(rmse_n_restricted)
            rmse_n_unrestricted = np.sqrt(mean_squared_error(self.unrestricted_data.true_clean[i], self.unrestricted_data.pred_clean[i]))
            unrestricted_list.append(rmse_n_unrestricted)
        return np.array(restricted_list), np.array(unrestricted_list)

    # Functions that DO require n
    def for_n(self, n):
        """
        Returns two CannonInstance objects for the given n (restricted, unrestricted).
        """
        return (
            CannonInstance(self.restricted_data, n),
            CannonInstance(self.unrestricted_data, n)
        )

    def compare_spectra(self, n):
        restricted, unrestricted = self.for_n(n)
        fig, ax = plt.subplots(3,1,figsize=(15,7))
        ax[0].plot(restricted.wav_real, restricted.s_real, 'gray', linewidth=5, alpha=0.5, label='True Spectrum')
        ax[0].plot(restricted.wav_pred, restricted.s_pred, 'b', linewidth=0.5, label='Restricted Model')
        ax[0].plot(unrestricted.wav_pred, unrestricted.s_pred, 'r', linewidth=0.5, label='Unrestricted Model')
        ax[0].set_xlim(3700,4200)
        ax[1].plot(restricted.wav_real, restricted.s_real, 'gray', linewidth=5, alpha=0.5)
        ax[1].plot(restricted.wav_pred, restricted.s_pred, 'b', linewidth=0.5)
        ax[1].plot(unrestricted.wav_pred, unrestricted.s_pred, 'r', linewidth=0.5)
        ax[1].set_xlim(4200,4700)
        ax[2].plot(restricted.wav_real, restricted.s_real, 'gray', linewidth=5, alpha=0.5)
        ax[2].plot(restricted.wav_pred, restricted.s_pred, 'b', linewidth=0.5)
        ax[2].plot(unrestricted.wav_pred, unrestricted.s_pred, 'r', linewidth=0.5)
        ax[2].set_xlim(4700,5200)
        fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.96])
        fig.suptitle(f"Retrieved Spectra Comparison (n={n})")
        fig.supxlabel('Wavelength (Angstroms)')
        fig.supylabel('Flux')
        fig.legend(loc='lower right', bbox_to_anchor=(0.95, 0.11), ncol=3)
        return None

    def compare_sfh(self, n):
        restricted, unrestricted = self.for_n(n)
        real_weights = restricted.data.f_real[restricted.global_idx]
        pred_restricted = restricted.data.pred_clean[restricted.n]
        pred_unrestricted = unrestricted.data.pred_clean[unrestricted.n]
        plt.figure(figsize=(10, 5))
        plt.bar(self.bin_centers, real_weights, width=self.bin_widths, align='center', color='gray', alpha=0.5, edgecolor='k', label='Real SFH')
        plt.bar(self.bin_centers, pred_restricted, width=self.bin_widths, align='center', color='b', alpha=0.2, edgecolor='b', label='Restricted Model')
        plt.bar(self.bin_centers, pred_unrestricted, width=self.bin_widths, align='center', color='r', alpha=0.2, edgecolor='r', label='Unrestricted Model')
        plt.xlabel('log(Age)')
        plt.ylabel('Weight')
        plt.xlim(2.5,10.5)
        plt.legend()
        plt.title(f"Retrieved SFH Comparison (n={n})")
        plt.show()
        return None

if __name__ == "__main__":
    pass