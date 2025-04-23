# kernel_cross_rho.py

import numpy as np
import scipy
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

def kernel_cross_rho(vis, msi, sigma=55.0, rank=True):
    """
    Distance-weighted rank (or Pearson) correlation between Visium genes
    and MSI m/z features, *without* one-to-one pixel matching.

    Parameters
    ----------
    vis   : AnnData  (n_vis_spots × n_genes)
    msi   : AnnData  (n_msi_pixels × n_mz)
    sigma : float    Gaussian bandwidth in same units as `vis.obsm['spatial']`
    rank  : bool     True → Spearman-like (ranked), False → Pearson

    Returns
    -------
    R     : ndarray  (n_genes × n_mz)   weighted correlation ρ
    """

    # ---------- 1. Gaussian weight matrix ----------
    X, Y = vis.obsm["spatial"], msi.obsm["spatial"]
    W = np.exp(-cdist(X, Y) ** 2 / (2.0 * sigma ** 2))
    W /= W.sum()  # so Σ_ij W_ij = 1

    # ---------- 2. Dense expression matrices ----------
    G = vis.X.toarray() if scipy.sparse.issparse(vis.X) else np.asarray(vis.X)
    Z = msi.X.toarray() if scipy.sparse.issparse(msi.X) else np.asarray(msi.X)

    # ---------- 3. Per-feature ranking (optional) ----------
    if rank:
        G = np.apply_along_axis(rankdata, 0, G)
        Z = np.apply_along_axis(rankdata, 0, Z)

    # ---------- 4. Weighted mean-centering ----------
    w_vis = W.sum(axis=1)
    w_msi = W.sum(axis=0)

    g_mean = w_vis @ G
    z_mean = w_msi @ Z

    Gc = G - g_mean
    Zc = Z - z_mean

    # ---------- 5. Numerator (weighted covariance) ----------
    num = Gc.T @ (W @ Zc)

    # ---------- 6. Denominator (√var_g · var_z) ----------
    var_g = w_vis @ (Gc ** 2)
    var_z = w_msi @ (Zc ** 2)
    den = np.sqrt(var_g[:, None] * var_z[None, :])

    return num / den