import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sfh import SFH

class AnalyseCannon:

    def __init__(self, base, n, restricted=False, use_intersection=True):
        """
        base: base filename
        n: index of spectrum to plot
        restricted: use restricted predictions
        use_intersection: if True, use intersection of valid indices from both restricted and unrestricted predictions (if both files exist)
        """
        import os
        self.restricted = restricted
        self.n = n
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

        self.pred_clean = self.f_pred[valid_indices]
        self.true_clean = self.f_real[valid_indices]

        self.sfh_real = SFH(self.true_clean[self.n])
        self.sfh_pred = SFH(self.pred_clean[self.n])

        self.wav_real, self.s_real, *_ = self.sfh_real.final_spectrum()
        self.wav_pred, self.s_pred, *_ = self.sfh_pred.final_spectrum()

        return None

    def plot_spectra(self):

        fig, ax = plt.subplots(4,1,figsize=(20,7))

        ax[0].plot(self.wav_real, self.s_real, 'r', alpha=0.5)
        ax[0].plot(self.wav_pred, self.s_pred, 'k', linewidth=0.5)
        ax[0].set_ylim(-0.15,0.1)
        ax[0].set_xlim(3500,4500)

        ax[1].plot(self.wav_real, self.s_real, 'r', alpha=0.5)
        ax[1].plot(self.wav_pred, self.s_pred, 'k', linewidth=0.5)
        ax[1].set_ylim(-0.1,0.05)
        ax[1].set_xlim(4500,5500)

        ax[2].plot(self.wav_real, self.s_real, 'r', alpha=0.5)
        ax[2].plot(self.wav_pred, self.s_pred, 'k', linewidth=0.5)
        ax[2].set_ylim(-0.02,0.01)
        ax[2].set_xlim(5500,6500)

        ax[3].plot(self.wav_real, self.s_real, 'r', alpha=0.5)
        ax[3].plot(self.wav_pred, self.s_pred, 'k', linewidth=0.5)
        ax[3].set_ylim(-0.15,0.05)
        ax[3].set_xlim(6500,7500)

        fig.suptitle(f"Spectra Comparison - {'Restricted' if self.restricted else 'Unrestricted'}")

        return None
    
    def plot_sfh(self):

        bin_arr = np.r_[np.array([0, 0.1, 20, 50, 100, 200, 500])*1e6, np.logspace(9.5, 10.15, 4)]
        binning = np.log10(bin_arr)
        bin_widths = np.diff(binning)
        self.bin_centers = binning[:-1] + bin_widths/2

        real_weights = self.true_clean[self.n]
        pred_weights = self.pred_clean[self.n]

        plt.figure(figsize=(10, 5))
        plt.bar(self.bin_centers, real_weights, width=bin_widths, align='center', color='b', alpha=0.5, edgecolor='b', label='f_real')
        plt.bar(self.bin_centers, pred_weights, width=bin_widths, align='center', color='r', alpha=0.5, edgecolor='r', label='f_pred')
        plt.xlabel('log(Age)')
        plt.ylabel('Weight')
        plt.xlim(4.5,10.5)
        plt.legend()
        plt.title(f"SFH Comparison - {'Restricted' if self.restricted else 'Unrestricted'}")
        plt.show()

        return None
    
    def rmse(self):

        rmse = np.sqrt(mean_squared_error(self.true_clean, self.pred_clean, multioutput='raw_values'))
        #for j, val in enumerate(rmse):
            #print(f"RMSE for label {j+1}: {val:.4f}")

        overall_rmse = np.sqrt(mean_squared_error(self.true_clean, self.pred_clean))
        #print(f"Overall RMSE: {overall_rmse:.4f}")

        return rmse, overall_rmse

    def plot_difference(self):

        diff = self.pred_clean - self.true_clean
        median = np.median(diff, axis=0)
        sixteen = np.percentile(diff, 16, axis=0)
        eightyfour = np.percentile(diff, 84, axis=0)

        plt.figure(figsize=(10,5))
        plt.errorbar(self.bin_centers, median, yerr=[median - sixteen, eightyfour - median], fmt='o', capsize=5)
        plt.xlabel('log(Age)')
        plt.ylabel('Difference (pred - true)')
        plt.xlim(4.5,10.5)
        plt.title(f"SFH Difference - {'Restricted' if self.restricted else 'Unrestricted'}")
        plt.show()

        return None

if __name__ == "__main__":
    pass