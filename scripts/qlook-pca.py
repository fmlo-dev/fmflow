# standard library
from pathlib import Path
from sys import argv


# dependencies
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

# constants
IS_OFF = (np.repeat(np.arange(100), 100) % 2).astype(bool)
T_AMB = 273.0  # K
HELP = "$ python qlook-pca.py /path/to/fits chbin"


if __name__ == "__main__":
    if len(argv) != 3:
        raise RuntimeError(HELP)
    else:
        hdu = fits.open(path := Path(argv[1]).resolve())[1]
        chbin = int(argv[2])

    for arrayid in tqdm(np.unique(hdu.data.arrayid)):
        sub = hdu.data[hdu.data.arrayid == arrayid]

        data = sub.arraydata
        data = data.reshape([data.shape[0], data.shape[1] // chbin, chbin]).mean(2)
        data = data[:, 100:-100]

        on = data[sub.scantype == "ON"][~IS_OFF]
        off = data[sub.scantype == "ON"][IS_OFF]
        r = data[sub.scantype == "R"].mean(0)
        sky = data[sub.scantype == "OFF"].mean(0)
        ch = np.arange(data.shape[1])

        # linear baseline
        T_linear = T_AMB * (on.mean(0) - off.mean(0)) / (r - sky)
        T_linear -= np.polyval(np.polyfit(ch, T_linear, 1), ch)

        # PCA baseline
        model = TruncatedSVD(5)
        model.fit(off)
        off_model = model.transform(on) @ model.components_
        T_pca = T_AMB * (on - off_model).mean(0) / (r - sky)

        # plotting
        fig, axes = plt.subplots(2, 1, sharex=True)

        ax = axes[0]
        ax.set_title(f"{path.name} / {arrayid}")
        ax.plot(ch, T_linear, label="Linear baseline")
        ax.axhspan(-T_linear.std(), +T_linear.std(), alpha=0.25)

        ax = axes[1]
        ax.plot(ch, T_pca, label="PCA baseline")
        ax.axhspan(-T_pca.std(), +T_pca.std(), alpha=0.25)

        for ax in axes:
            ax.set_xlabel(f"Channel ({chbin}-ch binning)")
            ax.set_ylabel("Ta* (K)")
            ax.set_ylim(
                np.min([T_linear, T_pca]),
                np.max([T_linear, T_pca]),
            )
            ax.grid(True)
            ax.legend(loc="upper right")

        fig.tight_layout()
        fig.savefig(path.with_suffix(f".pca.{arrayid}.pdf"), dpi=200)
