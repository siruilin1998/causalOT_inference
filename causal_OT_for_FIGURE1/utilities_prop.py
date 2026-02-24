import numpy as np
from collections import defaultdict
from utilities import cluster_z_values, empirical_distribution, ot_compute

def generate_data_prop(
        N: int,
        b0: float,
        b1: float,
        e_z,
):
    #Generate covariate Z ~ N(0, 1)
    Z = np.random.normal(loc=0.0, scale=1.0, size=N)
    Y0 = b0 * Z + np.random.normal(0, 1, size=N)
    Y1 = b1 * Z + np.random.normal(0, 1, size=N)

    
    # Compute propensity scores
    e_values = e_z(Z)
    
    #Assign treatment W ~ Bernoulli(e(Z))
    W = np.random.binomial(n=1, p=e_values)
    
    
    # Step 4: Separate Z into treatment and control groups
    Z_treat = Z[W == 1]
    Y_treat = Y1[W == 1]
    A1 = np.hstack((Y_treat.reshape(-1,1), Z_treat.reshape(-1,1)))
    
    Z_control = Z[W == 0]
    Y_control = Y0[W == 0]
    A0 = np.hstack((Y_control.reshape(-1,1), Z_control.reshape(-1,1)))
    
    Y = np.where(W == 1, Y1, Y0)
    
    
    data = {}
    
    data['X'] = Z
    
    data['W'] = W
    
    data['y'] = Y
    
    data['pis'] = e_values
    
    return A0, A1, data

def generate_data_square_prop(
        N: int,
        b0: float,
        b1: float,
        e_z,
):
    #Generate covariate Z ~ N(0, 1)
    Z = np.random.normal(loc=0.0, scale=1.0, size=N)
    Y0 = b0 * Z ** 2 + np.random.normal(0, 1, size=N)
    Y1 = b1 * Z ** 2 + np.random.normal(0, 1, size=N)

    
    # Compute propensity scores
    e_values = e_z(Z)
    
    #Assign treatment W ~ Bernoulli(e(Z))
    W = np.random.binomial(n=1, p=e_values)
    
    
    # Step 4: Separate Z into treatment and control groups
    Z_treat = Z[W == 1]
    Y_treat = Y1[W == 1]
    A1 = np.hstack((Y_treat.reshape(-1,1), Z_treat.reshape(-1,1)))
    
    Z_control = Z[W == 0]
    Y_control = Y0[W == 0]
    A0 = np.hstack((Y_control.reshape(-1,1), Z_control.reshape(-1,1)))
    
    Y = np.where(W == 1, Y1, Y0)
    
    
    data = {}
    
    data['X'] = Z
    
    data['W'] = W
    
    data['y'] = Y
    
    data['pis'] = e_values
    
    return A0, A1, data


def generate_data_prod_prop(
        N: int,
        e_z,
        b0: float,
        b1: float, 
        k0: float = None,
        k1: float = None,
        S0: float = None,
        S1: float = None,
    ):
    """
    Generate N samples of (y1, y2, z) using:
        y0 = (b0 z + k0) odot e0, z~N(0,Ip), e0~N(0,S1), z, e0 are column vector in this formula.
        y1 = (b1 z + k1) odot e1, z~N(0,Ip), e1~N(0,S2).

    Returns
    -------
    A Numpy array containing three NumPy arrays: y1, y2, and z.

    """   
    
    #Generate covariate Z ~ N(0, 1)
    Z = np.random.normal(loc=0.0, scale=1.0, size=N)
    Y0 = (b0 * Z + k0) * np.random.normal(0, 1, size=N)
    Y1 = (b1 * Z + k1) * np.random.normal(0, 1, size=N)

    
    # Compute propensity scores
    e_values = e_z(Z)
    
    #Assign treatment W ~ Bernoulli(e(Z))
    W = np.random.binomial(n=1, p=e_values)
    
    
    # Step 4: Separate Z into treatment and control groups
    Z_treat = Z[W == 1]
    Y_treat = Y1[W == 1]
    A1 = np.hstack((Y_treat.reshape(-1,1), Z_treat.reshape(-1,1)))
    
    Z_control = Z[W == 0]
    Y_control = Y0[W == 0]
    A0 = np.hstack((Y_control.reshape(-1,1), Z_control.reshape(-1,1)))
    
    Y = np.where(W == 1, Y1, Y0)
    
    
    data = {}
    
    data['X'] = Z
    
    data['W'] = W
    
    data['y'] = Y
    
    data['pis'] = e_values
    
    return A0, A1, data
    

    return np.hstack((Y0, Y1, Z))


def empirical_distribution_prop(data, treat, e_z):
    """
    Calculates the empirical distribution of the data.

    Args:
        data: A 1D numpy array of data values.

    Returns:
        A tuple containing:
            - support: A 1D numpy array of unique data values (support).
            - weights: A 1D numpy array of corresponding weights (frequencies).
    """
    support, counts = np.unique(data, return_counts=True)
    
    if treat:
        weights = counts * e_z(support) ** (-1)
    else:
        weights = counts * (1 - e_z(support)) ** (-1)
    
    weights /= sum(weights)
    
    return support, weights


def calculate_empirical_distributions_prop(clustered_data, treat, e_z):
    """
    Calculates empirical distributions for Z and Y|Z.

    Args:
        clustered_data: A NumPy array of shape (n_samples, 2) with clustered z-values.
                          First column: Y-values
                          Second column: Clustered z-values


    Returns:
        A tuple containing:
            - z_support: Support of Z's empirical distribution
            - z_weights: Weights of Z's empirical distribution
            - conditional_y_distributions: Dictionary storing empirical distributions for Y given each z in z_support
    """

    z_values = clustered_data[:, 1]
    y_values = clustered_data[:, 0]

    z_support, z_weights = empirical_distribution_prop(z_values, treat, e_z)

    conditional_y_distributions = defaultdict(lambda: (np.array([]), np.array([])))

    for z in z_support:
        indices = np.where(z_values == z)
        corresponding_y = y_values[indices]
        y_support, y_weights = empirical_distribution(corresponding_y)
        conditional_y_distributions[z] = (y_support, y_weights)

    return z_support, z_weights, conditional_y_distributions


def cot_estimator_prop(control_data, treatment_data, e_z, c=1.):
    """
    Estimates the Conditional Optimal Transport (COT) distance between two datasets.
    """

    clustered_control_data = cluster_z_values(control_data, c)
    clustered_treatment_data = cluster_z_values(treatment_data, c)

    # Step 1: Compute empirical distributions for each dataset
    control_z_support, control_z_weights, control_conditional_y = calculate_empirical_distributions_prop(clustered_control_data, False, e_z)
    treatment_z_support, treatment_z_weights, treatment_conditional_y = calculate_empirical_distributions_prop(clustered_treatment_data, True, e_z)

    # Step 2: Optimal transport matrix (L) between Z distributions
    distZ_control = (control_z_support, control_z_weights)
    distZ_treatment = (treatment_z_support, treatment_z_weights)
    _, L = ot_compute(distZ_control, distZ_treatment, 1)

    # Step 3 & 4: Compute transport cost matrix M and the final COT estimate
    M = np.zeros((len(control_z_support), len(treatment_z_support)))
    for i, z_c in enumerate(control_z_support):
        for j, z_t in enumerate(treatment_z_support):
            ot_d, _ = ot_compute(control_conditional_y[z_c], treatment_conditional_y[z_t], 2)
            M[i, j] = ot_d

    cot_estimate = np.sum(M * L)
    return cot_estimate
    