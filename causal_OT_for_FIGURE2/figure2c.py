from matplotlib import pyplot as plt
import numpy as np
from wavelet import SmoothConditionalWasserstein, closed_form_cw_gaussian_dz2, closed_form_cw_gaussian_dz2_prod
from otcompute import vip_estimate

def d2_generate_data(dy, dz, n, m, rng):
    a = np.array([0.12, 0.08])
    b = np.array([0.40,  0.50])   
            
    # Different covariances
    Sigma0 = np.array([[0.05**2, 0.0],
                    [0.0,     0.03**2]])

    return a, b, Sigma0 

def generate_data(dy, dz, n, m, rng):
    # Z ~ Uniform([0,1]^2)
    Z0 = rng.random((n, dz))
    Z1 = rng.random((m, dz))
    
    if dy == 2 and dz == 2:
        a, b, Sigma0 = d2_generate_data(dy, dz, n, m, rng)

    def mu0(z):
        # z: (N,2)
        return z @ a

    def delta_mu(z):
        return z @ b

    def clip01(x):
        return np.clip(x, 0.0, 1.0)

    # Sample
    Y0 = clip01(
        0.46 + mu0(Z0)[:, None] * rng.multivariate_normal(np.zeros(dy), Sigma0, size=n)
    )
    Y1 = clip01(
        0.5 + (mu0(Z1) + delta_mu(Z1))[:, None] *
         rng.multivariate_normal(np.zeros(dy), Sigma0, size=m)
    )

    return Y0, Z0, Y1, Z1, a, b, Sigma0

def iteration(dy, dz, N, rng, etalist, pp=False):
    # Data generation
    Y0, Z0, Y1, Z1, a, b, Sigma0 = generate_data(dy, dz, N, N, rng)

    # Closed-form CW (Gaussian, unclipped)
    cw_closed = closed_form_cw_gaussian_dz2_prod(
        a=a, b=b, Sigma0=Sigma0
    )

    # Wavelet CW
    scw = SmoothConditionalWasserstein(
        dy=dy,
        dz=dz,
        J_joint=4,   # (2^4)^(dy+dz)=16^4=65,536 cells
        J_z=6,       # (64)^2 for Z marginal
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
        Ny=600,
    )

    if pp:
        print("dy=2, dz=2, n=m=1000")
        print(f"Closed-form CW (Gaussian): {cw_closed:.6f}")
        print(f"Wavelet CW estimate:       {cw_wavelet:.6f}")
        print(f"Absolute error:            {abs(cw_wavelet - cw_closed):.6f}")
        print(f"Relative error:            {abs(cw_wavelet - cw_closed)/cw_closed:.3%}")

    # VIP estimation
    A0 = np.hstack([Y0, Z0])
    A1 = np.hstack([Y1, Z1])

    vip_estlist = vip_estimate(A0, A1, dy, dz, etalist)

    if pp:
        print("\nVIP estimates:")
        for eta, vip in zip(etalist, vip_estlist):
            print(f"Absolute error: eta={eta:.2f}: {vip:.6f}, {vip - cw_closed:.6f}")
            print(f"Relative error: eta={eta:.2f}: {vip:.6f}, {(vip - cw_closed) / cw_closed:.3%}")

    return np.array([cw_closed] + [cw_wavelet] + vip_estlist).reshape(1, -1)

def main(Nlist, etalist, dy, dz, seed, ITER=20):
    rng = np.random.default_rng(seed)
    res = {}
    for N in Nlist:
        print(f"\n=== N={N}: Iterating.. ===")
        res[N] = np.empty((0, 2 + len(etalist)))
        for it in range(ITER):
            if (it + 1) % 10 == 0:
                print(f"N={N}: Iteration {it+1}/{ITER}")
            results = iteration(dy, dz, N, rng, etalist, pp=False)
            res[N] = np.vstack([res[N], results]) if res[N].size else results

        print(f"\n=== N={N}: Summary ===")
        print(f"N={N}, dy={dy}, dz={dz}, seed={seed}, ITER={ITER}")
        mean_res = np.mean(res[N], axis=0)
        std_res = np.std(res[N], axis=0)
        print(f"Closed-form CW (Gaussian): Mean={mean_res[0]:.6f}, Std={std_res[0]:.6f}")
        print(f"Wavelet CW estimate:       Mean={mean_res[1]:.6f}, Std={std_res[1]:.6f}")
        for i in range(2, results.shape[1]):
            print(f"VIP estimate (eta={etalist[i-2]:.2f}): Mean={mean_res[i]:.6f}, Std={std_res[i]:.6f}")
    
    #plot results
    plot_results(Nlist, etalist, dy, dz, seed, res, ITER)
    plot_logresults(Nlist, etalist, dy, dz, seed, res, ITER)
    
def plot_results(Nlist, etalist, dy, dz, seed, res, ITER):
    import matplotlib.pyplot as plt 
    mean_res = np.vstack([np.mean(res[N], axis=0) for N in Nlist])
    plt.figure()
    # plt.plot(Nlist, np.abs(mean_res[:, 1] - mean_res[:, 0]), label="Wavelet CW estimate")
    plt.errorbar(Nlist, np.abs(mean_res[:, 1] - mean_res[:, 0]), yerr=[1.64 * np.std(res[N], axis=0)[1] / np.sqrt(ITER) for N in Nlist], fmt='o-', capsize=5, label="Wavelet CW estimate")
    for i in range(2, mean_res.shape[1]):
        # plt.plot(Nlist, np.abs(mean_res[:, i] - mean_res[:, 0]), label=f"VIP estimate (eta={etalist[i-2]:.2f})")
        plt.errorbar(Nlist, np.abs(mean_res[:, i] - mean_res[:, 0]), yerr=[1.64 * np.std(res[N], axis=0)[i] / np.sqrt(ITER) for N in Nlist], fmt='o-', capsize=5, label=rf"VIP estimate ($\eta$={etalist[i-2]:.0f})")
    plt.xlabel("N")
    plt.ylabel("Estimation Error")
    plt.legend()
    plt.title(f"Comparison of CW Estimates dy={dy}, dz={dz}")
    # plt.savefig(f"fig_3_dy{dy}_dz{dz}_seed{seed}.pdf")
    plt.show()

def plot_logresults(Nlist, etalist, dy, dz, seed, res, ITER):
    import matplotlib.pyplot as plt 
    mean_res = np.vstack([np.mean(res[N], axis=0) for N in Nlist])
    plt.figure()
    plt.errorbar(Nlist, np.abs(mean_res[:, 1] - mean_res[:, 0]), yerr=[1.64 * np.std(res[N], axis=0)[1] / np.sqrt(ITER) for N in Nlist], fmt='o-', capsize=5, label="Wavelet CW estimate")
    for i in range(2, mean_res.shape[1]):
        plt.errorbar(Nlist, np.abs(mean_res[:, i] - mean_res[:, 0]), yerr=[1.64 * np.std(res[N], axis=0)[i] / np.sqrt(ITER) for N in Nlist], fmt='o-', capsize=5, label=rf"VIP estimate ($\eta$={etalist[i-2]:.0f})")
    plt.xscale("log")
    plt.yscale('symlog', linthresh=1e-4)
    plt.xlabel("log N")
    plt.ylabel("Log Estimation Error")
    plt.legend()
    plt.title(f"Log Comparison of CW Estimates dy={dy}, dz={dz}")
    # plt.savefig(f"fig_3_log_dy{dy}_dz{dz}_seed{seed}.pdf")
    plt.show()

if __name__ == "__main__":
    x = np.logspace(np.log10(2000), np.log10(5000), num=6)
    x = np.round(x).astype(int)
    Nlist = x.tolist()

    etalist = [10]
    dy = 2
    dz = 2
    seed = 123
    main(Nlist, etalist, dy, dz, seed)

    #may be larger dy, dz can make the differences more visible