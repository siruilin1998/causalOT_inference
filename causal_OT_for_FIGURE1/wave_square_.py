# %% z square 1-dim
from truevalue import true_vc_square_wave
from utilities import generate_data_square_wave, treatment, cot_estimator
import numpy as np
from dualbounds.generic import DualBounds
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "../../")
import statistics
from wavelet import SmoothConditionalWasserstein


seed = 123
np.random.seed(seed)

b0 = np.array([[-0.04]])
b1 = np.array([[0.01]])
S0 = np.array([[0.001]])
S1 = np.array([[0.003]])
dy = 1
dz = 1

ITERSIZE = 200
sizelist = np.linspace(800, 2500, 6, dtype=int)


def main_cp(N, tvc, SIMSIZE, gap=True):
    ridge_vals = []
    knn_vals = []
    cot_vals = []

    for k in range(SIMSIZE):
        # generate raw data
        raw_data = generate_data_square_wave(N, b0, b1, S0, S1)

        ## generate post-treatment data
        A0, A1, data = treatment(raw_data, dy, dz)

        # Direct Vc estimate
        cot = cot_estimator(A0, A1)
        cot_vals.append(abs(cot - tvc) if gap else cot)

        ## Wavelet CW
        rng = np.random.default_rng(seed)
        scw = SmoothConditionalWasserstein(
            dy=dy,
            dz=dz,
            J_joint=4,
            J_z=6,
            wavelet="db4",
            mode="periodization",
            threshold=None,
            nonnegativity=True,
            renormalize=True,
        )

        Y0 = A0[:, :dy]
        Z0 = A0[:, dy:dy+dz]
        Y1 = A1[:, :dy]
        Z1 = A1[:, dy:dy+dz]

        cw_wavelet = scw.compute(
            Y0=Y0, Z0=Z0,
            Y1=Y1, Z1=Z1,
            rng=rng,
            Nz=200,
            Ny=500,
        )

        ## LL-ridge (wavelet proxy)
        LL_brd = cw_wavelet
        ridge_vals.append(abs(LL_brd - tvc) if gap else LL_brd)

        ## LL-knn estimate
        dbnd_knn = DualBounds(
            f=lambda y0, y1, x: (y0 - y1) ** 2,
            covariates=data['X'],
            treatment=data['W'],
            outcome=data['y'],
            propensities=data['pis'],
            outcome_model='knn',
        )

        result_knn = dbnd_knn.fit().results()
        LL_knn = result_knn.at['Estimate', 'Lower']
        knn_vals.append(abs(LL_knn - tvc) if gap else LL_knn)

        print(f'{k+1}/{SIMSIZE} complete.')

    def mean_and_se(x):
        x = np.asarray(x)
        return x.mean(), x.std(ddof=1) / np.sqrt(len(x))

    return {
        "ridge": mean_and_se(ridge_vals),
        "knn": mean_and_se(knn_vals),
        "cot": mean_and_se(cot_vals),
    }


# true Vc value
tvc = true_vc_square_wave(-b0, b1, S0, S1)

# main run
gap=True  #record estimation error (True) or estimation value (False)

ridge_mean, ridge_se = [], []
knn_mean, knn_se = [], []
cot_mean, cot_se = [], []

for sz in sizelist:
    res = main_cp(sz, tvc, SIMSIZE=ITERSIZE, gap=gap)

    m, s = res["ridge"]
    ridge_mean.append(m)
    ridge_se.append(s)

    m, s = res["knn"]
    knn_mean.append(m)
    knn_se.append(s)

    m, s = res["cot"]
    cot_mean.append(m)
    cot_se.append(s)

# convert to arrays (optional but clean)
ridge_mean, ridge_se = np.array(ridge_mean), np.array(ridge_se)
knn_mean, knn_se     = np.array(knn_mean), np.array(knn_se)
cot_mean, cot_se     = np.array(cot_mean), np.array(cot_se)

# plot
plt.figure(figsize=(7, 5))

plt.errorbar(
    sizelist, ridge_mean, yerr=ridge_se,
    marker='o', capsize=4, label='Wavelet-CW'
)

plt.errorbar(
    sizelist, knn_mean, yerr=knn_se,
    marker='s', capsize=4, label='LL-kNN'
)

plt.errorbar(
    sizelist, cot_mean, yerr=cot_se,
    marker='^', capsize=4, label='Plugin'
)

plt.xlabel("n")
plt.ylabel("Mean error" if gap else "Estimate")
plt.legend()
# plt.grid(alpha=0.3)
# plt.tight_layout()
plt.savefig("wave_error_quad_error.pdf")
plt.show()

