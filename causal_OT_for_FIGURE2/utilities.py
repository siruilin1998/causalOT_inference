import numpy as np
import numpy.typing as npt
from typing import Tuple
import pandas as pd

def generate_data(
        N: int,
        b0: npt.NDArray[np.float64],
        b1: npt.NDArray[np.float64], 
        S0: npt.NDArray[np.float64] = None,
        S1: npt.NDArray[np.float64] = None,
        a: npt.NDArray[np.float64] = None
    ) -> npt.NDArray[np.float64]:
    """
    Generate N samples of (y1, y2, z) using:
        y0 = b0 z + e0, z~N(0,Ip), e0~N(0,S1), z, e0 are column vector in this formula.
        y1 = b1 z + e1, z~N(0,Ip), e1~N(0,S2).

    Parameters
    ----------
    N : Sample size.
    b0 : A NumPy array of shape (dy, dz).
    b1 : A NumPy array of shape (dy, dz).
    S0 : A NumPy array of shape (dy, dy).
    S1 : A NumPy array of shape (dy, dy).

    Returns
    -------
    A Numpy array containing three NumPy arrays: y1, y2, and z.

    """
    
    if b0.shape != b1.shape:
        raise ValueError(f'The shape of b0 {b0.shape} does not match that of b1 {b1.shape}.')
    
    if S0 is not None:
        if not np.array_equal(S0.T, S0):
            raise ValueError('S0 needs to be symmetric matrix.')
        if b0.shape[0] != S0.shape[0]:
            raise ValueError(f'The first dim of b0:{b0.shape[0]} does not match the first dim of S0:{S0.shape[0]}.')
    else:
        if a is not None:
            S0 = np.eye(b0.shape[0]) * a
        else:
            S0 = np.eye(b0.shape[0])
    
    if S1 is not None:
        if not np.array_equal(S1.T, S1):
            raise ValueError('S1 needs to be symmetric matrix.')
        if b1.shape[0] != S1.shape[0]:
            raise ValueError(f'The first dim of b1:{b1.shape[0]} does not match the first dim of S1:{S1.shape[0]}.')
    else:
        if a is not None:
            S1 = np.eye(b1.shape[0]) * a
        else:
            S1 = np.eye(b1.shape[0])
    
    dy = b0.shape[0]
    dz = b1.shape[1]

    # Generate N samples of Z 
    Z = np.random.rand(N, dz)  
    
    # Generate N samples of e ~ N(0, S)
    # Cholesky decomposition of S to get L such that S = L * L^T
    L0 = np.linalg.cholesky(S0)  
    e0 = np.random.randn(N, dy) @ L0.T  
    # Compute Y = bZ + e for each row
    Y0 = Z @ b0.T + e0
    
    L1 = np.linalg.cholesky(S1)  
    e1 = np.random.randn(N, dy) @ L1.T  
    Y1 = Z @ b1.T + e1

    return np.hstack((Y0, Y1, Z))


def generate_data_uni(
        N: int,
        b0: npt.NDArray[np.float64],
        b1: npt.NDArray[np.float64], 
        seed1: int,
        seed2: int,
        S0: npt.NDArray[np.float64] = None,
        S1: npt.NDArray[np.float64] = None
    ) -> npt.NDArray[np.float64]:
    """
    Generate N samples of (y1, y2, z) using:
        y0 = b0 z + e0, z~N(0,Ip), e0~N(0,S1), z, e0 are column vector in this formula.
        y1 = b1 z + e1, z~N(0,Ip), e1~other noise.

    Parameters
    ----------
    N : Sample size.
    b0 : A NumPy array of shape (dy, dz).
    b1 : A NumPy array of shape (dy, dz).
    seed1: random seed for generating the mean of noise
    seed2: random seed for generating the noise itself.
    S0 : A NumPy array of shape (dy, dy).
    S1 : A NumPy array of shape (dy, dy).

    Returns
    -------
    A Numpy array containing three NumPy arrays: y1, y2, and z.

    """
    
    if b0.shape != b1.shape:
        raise ValueError(f'The shape of b0 {b0.shape} does not match that of b1 {b1.shape}.')
    
    if S0 is not None:
        if not np.array_equal(S0.T, S0):
            raise ValueError('S0 needs to be symmetric matrix.')
        if b0.shape[0] != S0.shape[0]:
            raise ValueError(f'The first dim of b0:{b0.shape[0]} does not match the first dim of S0:{S0.shape[0]}.')
    else:
        S0 = np.eye(b0.shape[0])
    
    if S1 is not None:
        if not np.array_equal(S1.T, S1):
            raise ValueError('S1 needs to be symmetric matrix.')
        if b1.shape[0] != S1.shape[0]:
            raise ValueError(f'The first dim of b1:{b1.shape[0]} does not match the first dim of S1:{S1.shape[0]}.')
    else:
        S1 = np.eye(b1.shape[0])
    
    dy = b0.shape[0]
    dz = b1.shape[1]

    # Generate N samples of Z 
    Z = np.random.randn(N, dz)  
    
    # Generate N samples of e ~ N(0, S)
    # Cholesky decomposition of S to get L such that S = L * L^T
    L0 = np.linalg.cholesky(S0)  
    e0 = np.random.randn(N, dy) @ L0.T  
    # Compute Y = bZ + e for each row
    Y0 = Z @ b0.T + e0
    
    L1 = np.linalg.cholesky(S1)  
    
    
    # e1 = np.random.rand(N, dy) @ L1.T  #uniform distribution
    
    e1 = []
    weights = np.array([1.,2.,3.])
    weights = weights / np.sum(weights)
    num_components = np.arange(len(weights))
    
    np.random.seed(seed1)
    means = []
    for _ in range(len(weights)):
        means.append(np.random.randn(b0.shape[0]))
    print(means)
    
    np.random.seed(seed2)
    for _ in range(N):
        # Choose a component based on weights
        component = np.random.choice(num_components, p=weights)
        
        # Sample from the chosen Gaussian
        sample = np.random.multivariate_normal(mean=means[component], cov=np.eye(b0.shape[0]))
        if len(e1):
            e1 = np.vstack((e1, sample.reshape(1, b0.shape[0])))
        else:
            e1 = sample.reshape(1, b0.shape[0])
    
    
    Y1 = Z @ b1.T + e1 @ L1.T 

    return np.hstack((Y0, Y1, Z))




def generate_data_square(
        N: int,
        b0: npt.NDArray[np.float64],
        b1: npt.NDArray[np.float64], 
        S0: npt.NDArray[np.float64] = None,
        S1: npt.NDArray[np.float64] = None
    ) -> npt.NDArray[np.float64]:
    """
    Generate N samples of (y1, y2, z) using:
        y0 = b0 z**2 + e0, z~N(0,Ip), e0~N(0,S1).
        y1 = b1 z**2 + e1, z~N(0,Ip), e1~N(0,S2).

    Parameters
    ----------
    N : Sample size.
    b0 : A NumPy array of shape (dy, dz).
    b1 : A NumPy array of shape (dy, dz).
    S0 : A NumPy array of shape (dy, dy).
    S1 : A NumPy array of shape (dy, dy).

    Returns
    -------
    A Numpy array containing three NumPy arrays: y1, y2, and z.

    """
    
    if b0.shape != b1.shape:
        raise ValueError(f'The shape of b0 {b0.shape} does not match that of b1 {b1.shape}.')
    
    if S0 is not None:
        if not np.array_equal(S0.T, S0):
            raise ValueError('S0 needs to be symmetric matrix.')
        if b0.shape[0] != S0.shape[0]:
            raise ValueError(f'The first dim of b0:{b0.shape[0]} does not match the first dim of S0:{S0.shape[0]}.')
    else:
        S0 = np.eye(b0.shape[0])
    
    if S1 is not None:
        if not np.array_equal(S1.T, S1):
            raise ValueError('S1 needs to be symmetric matrix.')
        if b1.shape[0] != S1.shape[0]:
            raise ValueError(f'The first dim of b1:{b1.shape[0]} does not match the first dim of S1:{S1.shape[0]}.')
    else:
        S1 = np.eye(b1.shape[0])
    
    dy = b0.shape[0]
    dz = b1.shape[1]

    # Generate N samples of Z 
    Z = np.random.randn(N, dz)  
    
    # Generate N samples of e ~ N(0, S)
    # Cholesky decomposition of S to get L such that S = L * L^T
    L0 = np.linalg.cholesky(S0)  
    e0 = np.random.randn(N, dy) @ L0.T  
    # Compute Y = bZ + e for each row
    Y0 = Z ** 2 @ b0.T + e0
    
    L1 = np.linalg.cholesky(S1)  
    e1 = np.random.randn(N, dy) @ L1.T  
    Y1 = Z ** 2 @ b1.T + e1

    return np.hstack((Y0, Y1, Z))



def generate_data_prod(
        N: int,
        b0: npt.NDArray[np.float64],
        b1: npt.NDArray[np.float64], 
        k0: npt.NDArray[np.float64] = None,
        k1: npt.NDArray[np.float64] = None,
        S0: npt.NDArray[np.float64] = None,
        S1: npt.NDArray[np.float64] = None,
    ) -> npt.NDArray[np.float64]:
    """
    Generate N samples of (y1, y2, z) using:
        y0 = (b0 z + k0) odot e0, z~N(0,Ip), e0~N(0,S1), z, e0 are column vector in this formula.
        y1 = (b1 z + k1) odot e1, z~N(0,Ip), e1~N(0,S2).

    Parameters
    ----------
    N : Sample size.
    b0 : A NumPy array of shape (dy, dz).
    b1 : A NumPy array of shape (dy, dz).
    k0 : A Numpy array of shape (dy, ).
    k1 : A Numpy array of shape (dy, ).
    S0 : A NumPy array of shape (dy, dy).
    S1 : A NumPy array of shape (dy, dy).

    Returns
    -------
    A Numpy array containing three NumPy arrays: y1, y2, and z.

    """
    
    if b0.shape != b1.shape:
        raise ValueError(f'The shape of b0 {b0.shape} does not match that of b1 {b1.shape}.')
    
    if S0 is not None:
        if not np.array_equal(S0.T, S0):
            raise ValueError('S0 needs to be symmetric matrix.')
        if b0.shape[0] != S0.shape[0]:
            raise ValueError(f'The first dim of b0:{b0.shape[0]} does not match the first dim of S0:{S0.shape[0]}.')
    else:
        S0 = np.eye(b0.shape[0])
    
    if S1 is not None:
        if not np.array_equal(S1.T, S1):
            raise ValueError('S1 needs to be symmetric matrix.')
        if b1.shape[0] != S1.shape[0]:
            raise ValueError(f'The first dim of b1:{b1.shape[0]} does not match the first dim of S1:{S1.shape[0]}.')
    else:
        S1 = np.eye(b1.shape[0])
    
    
    if k0 is not None:
        k0 = k0.flatten()
        if k0.shape[0] != b1.shape[0]:
            raise ValueError(f'The len of {k0.shape[0]} does not match dy {b1.shape[0]}.')
    else:
        k0 = np.ones(b1.shape[0])
    
    if k1 is not None:
        k1 = k1.flatten()
        if k1.shape[0] != b1.shape[0]:
            raise ValueError(f'The len of {k1.shape[0]} does not match dy {b1.shape[0]}.')
    else:
        k1 = np.ones(b1.shape[0])
    
    
    dy = b0.shape[0]
    dz = b1.shape[1]

    # Generate N samples of Z 
    Z = np.random.randn(N, dz)  
    
    # Generate N samples of e ~ N(0, S)
    # Cholesky decomposition of S to get L such that S = L * L^T
    L0 = np.linalg.cholesky(S0)  
    e0 = np.random.randn(N, dy) @ L0.T  
    # Compute Y = bZ + e for each row
    Y0 = (Z @ b0.T + np.vstack([k0.reshape(1,-1)] * N)) * e0
    
    L1 = np.linalg.cholesky(S1)  
    e1 = np.random.randn(N, dy) @ L1.T  
    Y1 = (Z @ b1.T + np.vstack([k1.reshape(1,-1)] * N)) * e1

    return np.hstack((Y0, Y1, Z))



def treatment(
        A : npt.NDArray[np.float64], 
        dy : int, 
        dz : int,
        pis: float = 0.5
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict]:
    """
    Given the synthetic data, generate randomized treatment and output the post-treatment data.    

    Parameters
    ----------
    A : npt.NDArray[np.float64]
        The whole data containing arrays of (y0, y1, z), z is covariate, y is outcome.
    dy : int
        Dim of outcome y.
    dz : int
        Dim of covariate Z.
    pis : float, optional.
        The propensity scores of W = 1 (treatment). The default is 0.5.

    Returns
    -------
    Two Numpy arrays containing (y1, z), (y2, z), and a dict: summary data.

    """
    if 2 * dy + dz != A.shape[1]:
        raise ValueError(f'The second dim of data A: {A.shape[1]} does not match (dy, dz): {(dy, dz)} for 2 * dy + dz.')
    
    
    # Randomly select the rows for treatment
    n_rows = A.shape[0]
    select_size = round(n_rows * pis)
    indices_1 = np.random.choice(n_rows, select_size, replace=False)
    indices_0 = np.array([i for i in range(n_rows) if i not in indices_1])
      
    # Assign 0 and 1 to the selected rows
    A_0 = A[indices_0]
    A_1 = A[indices_1]
    
    # Output the specified columns
    output_0 = np.hstack((A_0[:, :dy], A_0[:, 2 * dy:]))
    output_1 = np.hstack((A_1[:, dy: 2 * dy], A_1[:, 2 * dy:]))
    
    data = {}
    data['X'] = A[:, 2 * dy:] #covariate
    
    data['W'] = np.zeros(n_rows, dtype=int)
    data['W'][indices_1] = 1
    
    data['y'] = np.zeros((n_rows, dy))
    data['y'][indices_1] = A_1[:, dy: 2 * dy]
    data['y'][indices_0] = A_0[:, :dy]
    
    data['pis'] = np.ones(n_rows) * pis
    
    return output_0, output_1, data


def treatment_realdata(
        df: pd.DataFrame,
        outcome_col: str,
        covariate_col: str,
        treat_col: str,
        pis: float = 0.5,
        scale_covariate: float = 1.0
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict]:
    """
    Given the real data, generate randomized treatment and output the post-treatment data.    

    Parameters
    ----------
    df : pd.DataFrame
        Data file.
    outcome_col : str
        Col name of outcome.
    covariate_col : str
        Col name of covariates.
    treat_col : str
        Col name of treatment.
    pis : float, optional
        Propernsity score. The default is 0.5.
    scale_covariate : float, optional
        scaling parameter of covariate. The default is 1.0.
        
    Returns
    -------
    Two Numpy arrays containing (y1, z), (y2, z), and a dict: summary data.
    
    """
    
    data = {}
    data['X'] = df[[covariate_col]].to_numpy() * scale_covariate
    data['y'] = df[[outcome_col]].to_numpy()
    data['W'] = df[treat_col].astype(int).to_numpy()
    data['pis'] = pis
    
    treat_df = df[df[treat_col] == True].copy()
    control_df = df[df[treat_col] == False].copy()
    
    output_0 = control_df[[outcome_col, covariate_col]].copy()
    output_0.loc[:, covariate_col] *= scale_covariate
    output_0 = output_0.to_numpy()
    
    output_1 = treat_df[[outcome_col, covariate_col]].copy()
    output_1.loc[:, covariate_col] *= scale_covariate
    output_1 = output_1.to_numpy()
    
    return output_0, output_1, data

