# standard library
from pathlib import Path
from sys import argv


# dependencies
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import median_filter
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm


# constants
IS_OFF = (np.repeat(np.arange(100), 100) % 2).astype(bool)
T_AMB = 273.0  # K
HELP = "$ python qlook-godec.py /path/to/fits chbin"


def estimate_S(X, L, k=25, w=5):
    """Estimate sparse matrix (S)."""
    R = X - L
    spec = R.where(R.phi).sum("t")

    # smooth spectrum (optional)
    if w > 0:
        spec = xr.zeros_like(spec) + median_filter(spec, w)

    # k-th largest absolute values -> True
    spec = np.abs(spec)
    theta = (-spec).argsort().argsort() < k

    # calculate S and metadata (phi, theta)
    S = R * (R.phi * theta)
    S.coords["phi"] = R.phi
    S.coords["theta"] = theta

    return S


def estimate_L(X, S, r=5, seed=2021):
    """Estimate low-rank matrix (L)."""
    R = X - S
    R0 = R.mean("t")

    model = TruncatedSVD(r, random_state=seed)
    C = model.fit_transform(R - R0)
    P = model.components_
    L = xr.zeros_like(X) + C @ P + R0

    return L


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

        # GoDec baseline
        X = xr.DataArray(
            T_AMB * data[sub.scantype == "ON"] / (r - sky),
            dims=("t", "ch"),
            coords={"phi": ("t", ~IS_OFF)},
        )

        r = 10
        k = 15
        w = 5

        S = xr.zeros_like(X)

        for i in range(10):
            L = estimate_L(X, S, r=r)
            S = estimate_S(X, L, k=k, w=w)

        T_godec = (X - L).where(X.phi).mean("t").data

        # plotting
        fig, axes = plt.subplots(2, 1, sharex=True)

        ax = axes[0]
        ax.set_title(f"{path.name} / {arrayid}")
        ax.plot(ch, T_linear, label="Linear baseline")
        ax.axhspan(-T_linear.std(), +T_linear.std(), alpha=0.25)

        ax = axes[1]
        ax.plot(ch, T_godec, label="GoDec baseline")
        ax.axhspan(-T_godec.std(), +T_godec.std(), alpha=0.25)

        for ax in axes:
            ax.set_xlabel(f"Channel ({chbin}-ch binning)")
            ax.set_ylabel("Ta* (K)")
            ax.set_ylim(
                np.min([T_linear, T_godec]),
                np.max([T_linear, T_godec]),
            )
            ax.grid(True)
            ax.legend(loc="upper right")

        fig.tight_layout()
        fig.savefig(path.with_suffix(f".godec.{arrayid}.pdf"), dpi=200)
