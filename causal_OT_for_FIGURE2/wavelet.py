#@title Wavelet-based estimator

# """
# Smooth Conditional Wasserstein-2 distance via wavelet density estimators.

# Wavelet density estimator): implemented by
#   (i) [?]binning data on a dy+dz dimensional dyadic grid,
#   (ii) applying a multilevel wavelet transform (pywt.wavedecn),
#   (iii) optional hard/soft thresholding of detail coefficients,
#   (iv) inverse transform to get a smoothed density on the grid,
#   (v) optional projection to nonnegative + renormalization.

#   Remark:
#     True “boundary-corrected wavelets” are specialized basis functions.
#     Here we approximate boundary handling using pywt’s boundary modes (default: "periodization")
#     plus nonnegativity + renormalization. [?]In practice this is often fine.

# Conditional Wasserstein-2 distance: fit \hat P and \hat Q on (Y,Z), fit pooled marginal \hat R on Z,
#   sample Z~\hat R, then sample Y|Z=z from each conditional, and average W2^2.
# """

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import pywt
import ot

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import pywt
from scipy.linalg import sqrtm


# ----------------------------
# Helpers
# ----------------------------

def _soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)

def _hard_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    return x * (np.abs(x) >= lam)

def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x[:, None]
    if x.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {x.shape}.")
    return x

def _check_unit_cube(x: np.ndarray, name: str) -> None:
    if np.any(x < -1e-12) or np.any(x > 1 + 1e-12):
        raise ValueError(f"{name} must lie in [0,1] (elementwise). Got min={x.min()}, max={x.max()}.")


# ----------------------------
# Wavelet grid density estimator
# ----------------------------

@dataclass
class WaveletGridDensityEstimator:
    # """
    # Wavelet-smoothed density estimator on a dyadic grid over [0,1]^d.

    # Parameters
    # ----------
    # J : int
    #     Finest resolution; grid has (2^J) bins per dimension.
    # wavelet : str
    #     Wavelet family for pywt (e.g., "db4", "sym4", "coif1").
    # mode : str
    #     Boundary mode for pywt. "periodization" is typically best for densities on [0,1]^d.
    # threshold : Optional[float]
    #     Threshold lambda for wavelet detail coefficients.
    #     If None, no thresholding.
    # threshold_rule : str
    #     "hard" or "soft".
    # nonnegativity : bool
    #     Project negative values to 0.
    # renormalize : bool
    #     Renormalize so integral over [0,1]^d equals 1.
    # eps : float
    #     Small positive constant to avoid division by 0.
    # """
    J: int
    wavelet: str = "db4"
    mode: str = "periodization"
    threshold: Optional[float] = None
    threshold_rule: str = "soft"
    nonnegativity: bool = True
    renormalize: bool = True
    eps: float = 1e-12

    # learned
    d_: Optional[int] = None
    grid_shape_: Optional[Tuple[int, ...]] = None
    density_grid_: Optional[np.ndarray] = None
    bin_edges_: Optional[List[np.ndarray]] = None  # edges per dim

    def fit(self, X: np.ndarray) -> "WaveletGridDensityEstimator":
        """
        Fit density on [0,1]^d from samples X (n,d).
        """
        X = _as_2d(X)
        _check_unit_cube(X, "X")
        n, d = X.shape
        self.d_ = d

        # dyadic grid: 2^J bins per dimension
        B = 2 ** int(self.J)
        self.grid_shape_ = tuple([B] * d)

        # histogramdd gives probability mass per cell if density=False;
        # set density=True to get density w.r.t. Lebesgue (integrates to 1).
        hist, edges = np.histogramdd(X, bins=self.grid_shape_, range=[(0.0, 1.0)] * d, density=True)
        self.bin_edges_ = [np.asarray(e) for e in edges]

        # wavelet transform on the grid
        coeffs = pywt.wavedecn(hist, wavelet=self.wavelet, mode=self.mode, level=self.J)

        # optional thresholding of DETAIL coefficients (leave approximation coeffs[0] unchanged)
        if self.threshold is not None and self.threshold > 0:
            thr = float(self.threshold)
            for lvl in range(1, len(coeffs)):
                # coeffs[lvl] is a dict of detail arrays keyed by subband
                for k, arr in coeffs[lvl].items():
                    if self.threshold_rule.lower() == "hard":
                        coeffs[lvl][k] = _hard_threshold(arr, thr)
                    elif self.threshold_rule.lower() == "soft":
                        coeffs[lvl][k] = _soft_threshold(arr, thr)
                    else:
                        raise ValueError("threshold_rule must be 'hard' or 'soft'.")

        smoothed = pywt.waverecn(coeffs, wavelet=self.wavelet, mode=self.mode)

        # waverecn can return slightly larger due to padding; crop to grid
        slices = tuple(slice(0, s) for s in self.grid_shape_)
        smoothed = smoothed[slices]

        # enforce constraints
        if self.nonnegativity:
            smoothed = np.maximum(smoothed, 0.0)

        if self.renormalize:
            # Integral over [0,1]^d is approx sum(grid)*cell_volume
            cell_vol = (1.0 / (2 ** self.J)) ** d
            total_mass = float(np.sum(smoothed) * cell_vol)
            if total_mass <= self.eps:
                raise RuntimeError("Estimated density has ~0 total mass after projection; try smaller threshold.")
            smoothed = smoothed / total_mass

        self.density_grid_ = smoothed
        return self

    def density_on_grid(self) -> np.ndarray:
        if self.density_grid_ is None:
            raise RuntimeError("Call fit() first.")
        return self.density_grid_

    def sample(self, N: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        # """
        # Sample approximately from the fitted density by multinomial sampling on grid cells,
        # then uniform jitter within each selected cell.
        # """
        if self.density_grid_ is None or self.bin_edges_ is None or self.d_ is None:
            raise RuntimeError("Call fit() first.")
        rng = rng or np.random.default_rng()

        d = self.d_
        B = 2 ** self.J
        cell_vol = (1.0 / B) ** d

        p = (self.density_grid_.ravel() * cell_vol).astype(float)
        p_sum = p.sum()
        if not np.isfinite(p_sum) or p_sum <= 0:
            raise RuntimeError("Invalid probability mass from density grid.")
        p = p / p_sum

        idx = rng.choice(p.size, size=int(N), replace=True, p=p)
        multi_idx = np.array(np.unravel_index(idx, self.density_grid_.shape)).T  # (N,d)

        # jitter uniformly inside each cell using edges
        X = np.empty((N, d), dtype=float)
        for j in range(d):
            e = self.bin_edges_[j]
            lo = e[multi_idx[:, j]]
            hi = e[multi_idx[:, j] + 1]
            X[:, j] = lo + (hi - lo) * rng.random(N)
        return X

    def conditional_sampler(
        self,
        dy: int,
        dz: int,
        rng: Optional[np.random.Generator] = None,
    ) -> ConditionalSamplerFromJointGrid:
        # """
        # Return an object that can sample Y|Z=z from the joint density grid on (Y,Z).

        # Assumes X = (Y,Z) with dims d = dy+dz and coordinates ordered as [Y..., Z...].
        # """
        if self.density_grid_ is None or self.bin_edges_ is None or self.d_ is None:
            raise RuntimeError("Call fit() first.")
        if self.d_ != dy + dz:
            raise ValueError(f"Estimator dimension d={self.d_} but dy+dz={dy+dz}.")
        rng = rng or np.random.default_rng()
        return ConditionalSamplerFromJointGrid(
            density_grid=self.density_grid_,
            edges=self.bin_edges_,
            dy=dy,
            dz=dz,
            rng=rng,
        )


@dataclass
class ConditionalSamplerFromJointGrid:
    # """
    # [?]Sample from conditional Y|Z=z using a discretized joint density grid on (Y,Z).
    # """
    density_grid: np.ndarray
    edges: List[np.ndarray]
    dy: int
    dz: int
    rng: np.random.Generator

    def _z_to_index(self, z: np.ndarray) -> Tuple[int, ...]:
        # """
        # Map z in [0,1]^dz to nearest grid cell index along Z axes.
        # """
        z = np.asarray(z, dtype=float).reshape(-1)
        if z.size != self.dz:
            raise ValueError(f"z must have length dz={self.dz}, got {z.size}")
        if np.any(z < 0) or np.any(z > 1):
            raise ValueError("z must be in [0,1]^dz")

        idx = []
        for j in range(self.dz):
            e = self.edges[self.dy + j]
            # find bin index i such that e[i] <= z < e[i+1]
            i = int(np.searchsorted(e, z[j], side="right") - 1)
            i = max(0, min(i, len(e) - 2))
            idx.append(i)
        return tuple(idx)

    def sample_y_given_z(self, z: np.ndarray, Ny: int) -> np.ndarray:
        # """
        # Sample Ny draws from Y | Z = z using a slice of the joint grid at Z-bin(z),
        # then uniform jitter within the selected Y-cells.
        # """
        Ny = int(Ny)
        if Ny <= 0:
            raise ValueError("Ny must be positive.")
        z_idx = self._z_to_index(z)

        # Extract slice for fixed Z cell indices
        # density_grid has shape (B,...,B) for d=dy+dz
        # We fix the last dz indices to z_idx
        slicer = [slice(None)] * self.dy + list(z_idx)
        slice_density = self.density_grid[tuple(slicer)]  # shape (B,...,B) with dy dims

        # Convert density over Y-cells to probability mass:
        B = self.density_grid.shape[0]
        cell_vol_y = (1.0 / B) ** self.dy
        mass = (slice_density.ravel() * cell_vol_y).astype(float)
        s = mass.sum()
        if s <= 0 or not np.isfinite(s):
            # fallback: uniform over Y grid if conditional degenerates
            mass = np.ones_like(mass) / mass.size
        else:
            mass = mass / s

        idx = self.rng.choice(mass.size, size=Ny, replace=True, p=mass)
        y_multi_idx = np.array(np.unravel_index(idx, slice_density.shape)).T  # (Ny,dy)

        # jitter within Y-cells
        Y = np.empty((Ny, self.dy), dtype=float)
        for j in range(self.dy):
            e = self.edges[j]
            lo = e[y_multi_idx[:, j]]
            hi = e[y_multi_idx[:, j] + 1]
            Y[:, j] = lo + (hi - lo) * self.rng.random(Ny)
        return Y


# ----------------------------
# Smooth Conditional Wasserstein distance
# ----------------------------

def w2_squared_empirical(Y0: np.ndarray, Y1: np.ndarray) -> float:
    # """
    # Compute W2^2 between two empirical distributions with equal weights.
    # Uses:
    #   - exact 1D formula if dy=1 (sort+mean squared diff),
    #   - POT (ot.emd2) for dy>=2.

    # Y0: (Ny,dy), Y1: (Ny,dy)
    # """
    Y0 = _as_2d(Y0)
    Y1 = _as_2d(Y1)
    if Y0.shape != Y1.shape:
        raise ValueError("Y0 and Y1 must have the same shape for equal-weight W2.")
    Ny, dy = Y0.shape

    if dy == 1:
        a = np.sort(Y0[:, 0])
        b = np.sort(Y1[:, 0])
        return float(np.mean((a - b) ** 2))

    if ot is None:
        raise ImportError("POT (ot) is required for dy>=2. Install with: pip install pot")

    # uniform weights
    w = np.full(Ny, 1.0 / Ny)
    # squared Euclidean cost matrix
    M = ot.dist(Y0, Y1, metric="euclidean") ** 2
    return float(ot.emd2(w, w, M))


@dataclass
class SmoothConditionalWasserstein:
    dy: int
    dz: int
    J_joint: int = 5
    J_z: int = 6
    wavelet: str = "db4"
    mode: str = "periodization"
    threshold: Optional[float] = None
    threshold_rule: str = "soft"
    nonnegativity: bool = True
    renormalize: bool = True

    def compute(
        self,
        Y0: np.ndarray,
        Z0: np.ndarray,
        Y1: np.ndarray,
        Z1: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        Nz: Optional[int] = None,
        Ny: Optional[int] = None,
    ) -> float:
        # """
        # Input:
        #     (Y0,Z0) ~ P, size n
        #     (Y1,Z1) ~ Q, size m
        # Output:
        #     \hat{CW}(\tilde P, \tilde Q)
        # """
        rng = rng or np.random.default_rng()

        Y0 = _as_2d(Y0); Z0 = _as_2d(Z0)
        Y1 = _as_2d(Y1); Z1 = _as_2d(Z1)
        if Y0.shape[1] != self.dy or Y1.shape[1] != self.dy:
            raise ValueError("Y dims mismatch dy.")
        if Z0.shape[1] != self.dz or Z1.shape[1] != self.dz:
            raise ValueError("Z dims mismatch dz.")

        _check_unit_cube(Y0, "Y0"); _check_unit_cube(Z0, "Z0")
        _check_unit_cube(Y1, "Y1"); _check_unit_cube(Z1, "Z1")

        n = Y0.shape[0]
        m = Y1.shape[0]
        nm = max(n, m)

        if Nz is None:
            Nz = int(math.ceil(nm * math.log(max(nm, 2))))
        if Ny is None:
            expo = max(self.dy / 4.0, 1.0)
            Ny = int(math.ceil((nm ** expo) * math.log(max(nm, 2))))
            # cap Ny to avoid absurd sizes by default; adjust as needed
            Ny = min(Ny, 2000)

        # Estimate joint densities \hat P and \hat Q on (Y,Z)
        X0 = np.hstack([Y0, Z0])
        X1 = np.hstack([Y1, Z1])

        P_hat = WaveletGridDensityEstimator(
            J=self.J_joint,
            wavelet=self.wavelet,
            mode=self.mode,
            threshold=self.threshold,
            threshold_rule=self.threshold_rule,
            nonnegativity=self.nonnegativity,
            renormalize=self.renormalize,
        ).fit(X0) # [?]On grid

        Q_hat = WaveletGridDensityEstimator(
            J=self.J_joint,
            wavelet=self.wavelet,
            mode=self.mode,
            threshold=self.threshold,
            threshold_rule=self.threshold_rule,
            nonnegativity=self.nonnegativity,
            renormalize=self.renormalize,
        ).fit(X1)

        # Estimate pooled marginal \hat R on Z using pooled Z samples
        Z_pool = np.vstack([Z0, Z1])
        R_hat = WaveletGridDensityEstimator(
            J=self.J_z,
            wavelet=self.wavelet,
            mode=self.mode,
            threshold=self.threshold,
            threshold_rule=self.threshold_rule,
            nonnegativity=self.nonnegativity,
            renormalize=self.renormalize,
        ).fit(Z_pool)

        P_cond = P_hat.conditional_sampler(dy=self.dy, dz=self.dz, rng=rng)
        Q_cond = Q_hat.conditional_sampler(dy=self.dy, dz=self.dz, rng=rng)

        # Sample Z~\hat R
        Z_tilde = R_hat.sample(Nz, rng=rng)  # shape (Nz, dz)

        # For each z, sample Y|z from both and compute W2^2, then average
        w2_vals = []
        for t in range(Nz):
            z = Z_tilde[t, :]
            Yt0 = P_cond.sample_y_given_z(z=z, Ny=Ny)
            Yt1 = Q_cond.sample_y_given_z(z=z, Ny=Ny)
            w2_vals.append(w2_squared_empirical(Yt0, Yt1))

        return float(np.mean(w2_vals))


def closed_form_cw_gaussian_dz2(
    a: np.ndarray,
    B: np.ndarray,
    Sigma0: np.ndarray,
    Sigma1: np.ndarray,
) -> float:
    """
    Z ~ Unif([0,1]^2)
    mu1(z) - mu0(z) = a + B z
    """
    a = np.asarray(a, float).reshape(-1)
    B = np.asarray(B, float)

    # Mean part
    # mean_term = (
    #     a @ a
    #     + a @ (B @ np.ones(B.shape[1]))
    #     + (1.0 / 3.0) * np.sum(B**2)
    #     + 0.5 * (np.sum((B.T @ B)) - np.sum(np.diag(B.T @ B)))
    # )
    

    Ez = np.array([0.5, 0.5])
    Ezz = np.array([[1/3, 1/4],
                    [1/4, 1/3]])

    term1 = a @ a
    term2 = 2 * a @ B @ Ez
    term3 = np.trace(B.T @ B @ Ezz)
    mean_term = term1 + term2 + term3


    # Covariance part
    S1_half = sqrtm(Sigma1)
    middle = sqrtm(S1_half @ Sigma0 @ S1_half)
    cov_term = np.trace(Sigma0 + Sigma1 - 2 * middle)

    return float(mean_term + cov_term)

def closed_form_cw_gaussian_dy1dz2(
    a: np.ndarray,
    B: np.ndarray,
    Sigma0: np.ndarray,
    Sigma1: np.ndarray,
) -> float:
    """
    Z ~ Unif([0,1]^2)
    mu1(z) - mu0(z) = a + B z in dim 1
    """
    a = np.asarray(a, float).reshape(-1)
    B = np.asarray(B, float)

    # Mean part
    # mean_term = (
    #     a @ a
    #     + a @ (B @ np.ones(B.shape[1]))
    #     + (1.0 / 3.0) * np.sum(B**2)
    #     + 0.5 * (np.sum((B.T @ B)) - np.sum(np.diag(B.T @ B)))
    # )

    Ez = np.array([0.5, 0.5])
    Ezz = np.array([[1/3, 1/4],
                    [1/4, 1/3]])

    term1 = a @ a
    term2 = 2 * a @ B @ Ez
    term3 = np.trace(B.T @ B @ Ezz)
    mean_term = term1 + term2 + term3

    # Covariance part
    S1_half = sqrtm(Sigma1)
    middle = sqrtm(S1_half @ Sigma0 @ S1_half)
    cov_term = np.trace(Sigma0 + Sigma1 - 2 * middle)

    return float(mean_term + cov_term)

def closed_form_cw_gaussian_dz2_quad(
        a: float,
        b: float,
        Sigma0: np.ndarray, 
        Sigma1: np.ndarray,
) -> float:
    """
    Z ~ Unif([0,1]^2)
    mu0(z) = a * (z - 0.5)^2
    mu1(z) = b * (z - 0.5)^2
    """
    a = float(a)
    b = float(b)

    # Mean part
    mean_term = (
        (a - b) ** 2 / 40.0 # E[||(z - 0.5)**2||_2^2] = dz / 80
    )

    # Covariance part
    S1_half = sqrtm(Sigma1)
    middle = sqrtm(S1_half @ Sigma0 @ S1_half)
    cov_term = np.trace(Sigma0 + Sigma1 - 2 * middle)

    return float(mean_term + cov_term)


def closed_form_cw_gaussian_dz12_quad(
        a: float,
        b: float,
        Sigma0: np.ndarray, 
        Sigma1: np.ndarray,
) -> float:
    """
    Z ~ Unif([0,1]^2)
    mu0(z) = a @ (z - 0.5)^2, a in R^2
    mu1(z) = b @ (z - 0.5)^2, b in R^2
    """

    def exact(a):
        a1, a2 = a
        return (a1*a1 + a2*a2)/80 + (a1*a2)/72

    # Mean part
    mean_term = exact(a - b)

    # Covariance part
    S1_half = sqrtm(Sigma1)
    middle = sqrtm(S1_half @ Sigma0 @ S1_half)
    cov_term = np.trace(Sigma0 + Sigma1 - 2 * middle)

    return float(mean_term + cov_term)


def closed_form_cw_gaussian_dz2_prod(
        a: np.ndarray,
        b: np.ndarray,
        Sigma0: np.ndarray,
    ) -> float:
    """
    Z ~ Unif([0,1]^2)
    y0 = a @ z * N(0, Sigma0) 
    y1 = (a + b) @ z * N(0, Sigma0)
    """
    # Mean part
    mean_term = (
        (1.0 / 12.0) * np.dot(b, b) + 0.25 * (np.sum(b) ** 2) # E[||(z @ b)**2||_2^2] = (1/12) b^2 + (1/4)(sum b)^2
    )

    # Covariance part
    cov_term = np.trace(Sigma0)

    return float(mean_term * cov_term) + 0.04 ** 2