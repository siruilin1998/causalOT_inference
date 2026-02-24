import sys; sys.path.insert(0, "../../")
from dualbounds.generic import DualBounds

from matplotlib import pyplot as plt
import numpy as np
from wavelet import SmoothConditionalWasserstein, closed_form_cw_gaussian_dz2
from otcompute import vip_estimate
from utilities import generate_data_square_wave, treatment, cot_estimator

def d1_generate_data(dy, dz, n, m, rng):
    # Mean structure
    base  = np.array([0.35])
    slope = np.array([[0.20]])   # 1 x 1

    a = np.array([0.12])
    B = np.array([[0.10]])       # 1 x 1


    # Different covariances
    Sigma0 = np.array([[0.05**2]])
    Sigma1 = np.array([[0.07**2]])

    return base, slope, a, B, Sigma0, Sigma1


def generate_data(dy, dz, n, m, rng):
    # Z ~ Uniform([0,1]^2)
    Z0 = rng.random((n, dz))
    Z1 = rng.random((m, dz))
    
    if dy == 1 and dz == 1:
        base, slope, a, B, Sigma0, Sigma1 = d1_generate_data(dy, dz, n, m, rng)

    def mu0(z):
        # z: (N,2)
        return base[None, :] + z @ slope.T

    def delta_mu(z):
        return a[None, :] + z @ B.T

    def clip01(x):
        return np.clip(x, 0.0, 1.0)

    # Sample
    Y0 = clip01(
        mu0(Z0) + rng.multivariate_normal(np.zeros(dy), Sigma0, size=n)
    )
    Y1 = clip01(
        mu0(Z1) + delta_mu(Z1)
        + rng.multivariate_normal(np.zeros(dy), Sigma1, size=m)
    )

    return Y0, Z0, Y1, Z1, a, B, Sigma0, Sigma1 

def iteration(dy, dz, N, rng, pp=False):
    # Data generation
    Y0, Z0, Y1, Z1, a, B, Sigma0, Sigma1 = generate_data(dy, dz, N, N, rng)

    # Closed-form CW (Gaussian, unclipped)
    cw_closed = closed_form_cw_gaussian_dz2(
        a=a, B=B, Sigma0=Sigma0, Sigma1=Sigma1
    )

    # Wavelet CW
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

    cw_wavelet = scw.compute(
        Y0=Y0, Z0=Z0,
        Y1=Y1, Z1=Z1,
        rng=rng,
        Nz=100,
        Ny=300,
    )

    ## Plugin
    A0 = np.hstack([Y0.reshape(-1,1), Z0.reshape(-1, 1)])
    A1 = np.hstack([Y1.reshape(-1,1), Z1.reshape(-1, 1)])
    cot = cot_estimator(A0, A1)

    ## LL-knn estimate
    # ensure y is flat
    Y0 = Y0.reshape(-1)
    Y1 = Y1.reshape(-1)

    X = np.vstack([Z0, Z1])                         # covariates
    y = np.concatenate([Y0, Y1], axis=0)            # outcome
    W = np.concatenate([np.zeros(len(Y0)), np.ones(len(Y1))]).astype(int)
    pis = np.full(len(y), float(1/2))

    data = {"X": X, "y": y, "W": W, "pis": pis}

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

    return np.array([cw_closed] + [cw_wavelet] + [LL_knn, cot]).reshape(1, -1)

def main(Nlist, dy, dz, seed, ITER=300):
    rng = np.random.default_rng(seed)
    res = {}
    for N in Nlist:
        print(f"\n=== N={N}: Iterating.. ===")
        res[N] = np.empty((0, 4))
        for it in range(ITER):
            print(f"N={N}: Iteration {it+1}/{ITER}")
            results = iteration(dy, dz, N, rng, pp=False)
            res[N] = np.vstack([res[N], results]) if res[N].size else results

        print(f"\n=== N={N}: Summary ===")
        print(f"N={N}, dy={dy}, dz={dz}, seed={seed}, ITER={ITER}")
        mean_res = np.mean(res[N], axis=0)
        std_res = np.std(res[N], axis=0)
        print(f"Closed-form CW (Gaussian): Mean={mean_res[0]:.6f}, Std={std_res[0]:.6f}")
        print(f"Wavelet CW estimate:       Mean={mean_res[1]:.6f}, Std={std_res[1]:.6f}")
        print(f"LL-kNN estimate:          Mean={mean_res[2]:.6f}, Std={std_res[2]:.6f}")
        print(f"Plugin estimate:          Mean={mean_res[3]:.6f}, Std={std_res[3]:.6f}")

    #plot results
    plot_results(Nlist, res, ITER)
    
def plot_results(Nlist, res, ITER):
    import matplotlib.pyplot as plt 
    mean_res = np.vstack([np.mean(res[N], axis=0) for N in Nlist])
    plt.figure()
    plt.errorbar(Nlist, np.abs(mean_res[:, 1] - mean_res[:, 0]), yerr=[np.std(res[N], axis=0)[1] / np.sqrt(ITER) for N in Nlist], fmt='o-', capsize=5, label="Wavelet-CW")
    plt.errorbar(Nlist, np.abs(mean_res[:, 2] - mean_res[:, 0]), yerr=[np.std(res[N], axis=0)[1] / np.sqrt(ITER) for N in Nlist], fmt='o-', capsize=5, label="LL-kNN")
    plt.errorbar(Nlist, np.abs(mean_res[:, 3] - mean_res[:, 0]), yerr=[np.std(res[N], axis=0)[1] / np.sqrt(ITER) for N in Nlist], fmt='o-', capsize=5, label="Plugin")
    
    plt.xlabel("n")
    plt.ylabel("Estimation Error")
    plt.legend()
    plt.savefig(f"wave_linear_error.pdf")
    plt.show()

if __name__ == "__main__":
    dy = 1
    dz = 1
    Nlist = np.linspace(800, 2500, 6, dtype=int)
    seed = 123
    ITER = 100

    main(Nlist, dy, dz, seed, ITER)