import numpy as np
import numpy.typing as npt
from typing import Tuple
import pandas as pd

def generate_data(
        N: int,
        b0: npt.NDArray[np.float64],
        b1: npt.NDArray[np.float64], 
        S0: npt.NDArray[np.float64] = None,
        S1: npt.NDArray[np.float64] = None
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
    e1 = np.random.randn(N, dy) @ L1.T  
    Y1 = Z @ b1.T + e1

    return np.hstack((Y0, Y1, Z))

def generate_data_wave(
        N: int,
        b0: npt.NDArray[np.float64],
        b1: npt.NDArray[np.float64], 
        S0: npt.NDArray[np.float64],
        S1: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
    """
    Generate N samples of (y1, y2, z) using:
        y0 = b0 z + e0, z~U(0,Ip), e0~N(0,S1), z, e0 are column vector in this formula.
        y1 = b1 z + e1, z~U(0,Ip), e1~N(0,S2).

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
    
    # if b0.shape != b1.shape:
    #     raise ValueError(f'The shape of b0 {b0.shape} does not match that of b1 {b1.shape}.')
    
    # if S0 is not None:
    #     if not np.array_equal(S0.T, S0):
    #         raise ValueError('S0 needs to be symmetric matrix.')
    #     if b0.shape[0] != S0.shape[0]:
    #         raise ValueError(f'The first dim of b0:{b0.shape[0]} does not match the first dim of S0:{S0.shape[0]}.')
    # else:
    #     S0 = np.eye(b0.shape[0])
    
    # if S1 is not None:
    #     if not np.array_equal(S1.T, S1):
    #         raise ValueError('S1 needs to be symmetric matrix.')
    #     if b1.shape[0] != S1.shape[0]:
    #         raise ValueError(f'The first dim of b1:{b1.shape[0]} does not match the first dim of S1:{S1.shape[0]}.')
    # else:
    #     S1 = np.eye(b1.shape[0])
    
    dy = b0.shape[0]
    dz = b1.shape[1]

    # Generate N samples of Z 
    Z = np.random.rand(N, dz)  
    
    # Generate N samples of e ~ N(0, S)
    # Cholesky decomposition of S to get L such that S = L * L^T
    L0 = np.linalg.cholesky(S0)  
    e0 = np.random.randn(N, dy) @ L0.T  
    # Compute Y = bZ + e for each row
    Y0 = Z * b0 + e0
    
    L1 = np.linalg.cholesky(S1)  
    e1 = np.random.randn(N, dy) @ L1.T  
    Y1 = Z @ b1.T + e1

    def clip01(x):
        return np.clip(x, 0.0, 1.0)

    return np.hstack((clip01(Y0), clip01(Y1), Z))

def generate_data_perturb(
        N: int,
        eps: float,
        b0: npt.NDArray[np.float64],
        b1: npt.NDArray[np.float64], 
        S0: npt.NDArray[np.float64] = None,
        S1: npt.NDArray[np.float64] = None
    ) -> npt.NDArray[np.float64]:
    """
    Generate N samples of (y1, y2, z) using:
        y0 = b0 z + e0, z~N(0,Ip), e0~N(0,S1), z, e0 are column vector in this formula.
        y1 = b1 z + e1, z~N(0,Ip) + eps + eps * N(0, Ip), e1~N(0,S2).

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
    Y0 = Z @ b0.T + e0
    
    L1 = np.linalg.cholesky(S1)  
    e1 = np.random.randn(N, dy) @ L1.T  
    Z += eps * np.random.randn(N, dz) 
    Y1 = Z @ b1.T + e1

    return np.hstack((Y0, Y1, Z))

def generate_data_perturb_uni(
        N: int,
        eps: float,
        b0: npt.NDArray[np.float64],
        b1: npt.NDArray[np.float64], 
        S0: npt.NDArray[np.float64] = None,
        S1: npt.NDArray[np.float64] = None
    ) -> npt.NDArray[np.float64]:
    """
    Generate N samples of (y1, y2, z) using:
        y0 = b0 z + e0, z~N(0,Ip), e0~N(0,S1), z, e0 are column vector in this formula.
        y1 = b1 z + e1, z~N(0,Ip) + eps + eps * N(0, Ip), e1~N(0,S2).

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
    Z = np.random.rand(N, dz)  
    
    # Generate N samples of e ~ N(0, S)
    # Cholesky decomposition of S to get L such that S = L * L^T
    L0 = np.linalg.cholesky(S0)  
    e0 = np.random.randn(N, dy) @ L0.T  
    # Compute Y = bZ + e for each row
    Y0 = Z @ b0.T + e0
    
    L1 = np.linalg.cholesky(S1)  
    e1 = np.random.randn(N, dy) @ L1.T  
    Z += eps
    # Z += eps * np.random.rand(N, dz) 
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


def generate_data_square_wave(
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
    Z = np.random.rand(N, dz)  
    
    # Generate N samples of e ~ N(0, S)
    # Cholesky decomposition of S to get L such that S = L * L^T
    L0 = np.linalg.cholesky(S0)  
    e0 = np.random.randn(N, dy) @ L0.T  
    # Compute Y = bZ + e for each row
    Y0 = Z ** 2 @ b0.T + e0 + 0.5
    
    L1 = np.linalg.cholesky(S1)  
    e1 = np.random.randn(N, dy) @ L1.T  
    Y1 = Z ** 2 @ b1.T + e1 + 0.5

    def clip01(x):
        return np.clip(x, 0.0, 1.0)

    return np.hstack((clip01(Y0), clip01(Y1), Z))

def generate_data_square_perturb(
        N: int,
        eps: float,
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
    Z += eps * np.random.randn(N, dz)  
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

def generate_data_prod_wave(
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
    Z = np.random.rand(N, dz)  
    
    # Generate N samples of e ~ N(0, S)
    # Cholesky decomposition of S to get L such that S = L * L^T
    L0 = np.linalg.cholesky(S0)  
    e0 = np.random.randn(N, dy) @ L0.T  
    # Compute Y = bZ + e for each row
    Y0 = (Z @ b0.T + np.vstack([k0.reshape(1,-1)] * N)) * e0 + 0.5
    
    L1 = np.linalg.cholesky(S1)  
    e1 = np.random.randn(N, dy) @ L1.T  
    Y1 = (Z @ b1.T + np.vstack([k1.reshape(1,-1)] * N)) * e1 + 0.5

    def clip01(x):
        return np.clip(x, 0.0, 1.0)

    return np.hstack((clip01(Y0), clip01(Y1), Z))


def generate_data_prod_perturb(
        N: int,
        eps: float,
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
    Z += eps * np.random.randn(N, dz)  
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
    Two Numpy arrays containing (y0, z), (y1, z), and a dict: summary data.

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


def treatment_perturb(
        A : npt.NDArray[np.float64], 
        eps: float,
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
    Two Numpy arrays containing (y0, z), (y1, z), and a dict: summary data.

    """
    if 2 * dy + dz != A.shape[1]:
        raise ValueError(f'The second dim of data A: {A.shape[1]} does not match (dy, dz): {(dy, dz)} for 2 * dy + dz.')
    
    
    # Randomly select the rows for treatment
    n_rows = A.shape[0]
    select_size = round(n_rows * pis)
    indices_1 = np.random.choice(n_rows, select_size, replace=False)
    indices_0 = np.array([i for i in range(n_rows) if i not in indices_1])
      
    noise = np.random.rand(len(indices_1), A.shape[1] - 2 * dy)
    A[np.array(indices_1), 2 * dy:] += eps * noise
    
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


###### direct COT estimation

def COT_trueval(f0, f1, type_='normal', num_simulations=int(1e6)):
    """
    Suppose that 
    Y(0) = f0(Z(0)) + N(0, I),
    Y(1) = f1(Z(1)) + N(0, I).
    h = (y0 - y1)^2
    Computes COT by the expectation of E[(f0(Z) - f1(Z))^2] using Monte Carlo simulation.
      
    Args:
      f0:
      f1:
      num_simulations: The number of simulations to run.
      
    Returns:
      The estimated expectation.
    """
    if type_ == 'normal':
        z_samples = np.random.normal(0, 1, num_simulations)
    elif type_ == 'uniform':
        z_samples = np.random.uniform(0, 1, num_simulations)
    else:
        raise ValueError('The given type of distribution for Z is not recognized.')
    sum_squared_terms = 0
      
    for z in z_samples:
      sum_squared_terms += (f0(z) - f1(z))**2
      
    return sum_squared_terms / num_simulations


def new_generate_data(f0, f1, n_samples):
    """
    Y(0) = f0(Z(0)) + N(0, I), Z(0) ~ U[0,1],
    Y(1) = f1(Z(1)) + N(0, I), Z(1) ~ U[0,1].
    Generates synthetic data for the control and treatment groups.

    Args:
      n_samples: The number of samples to generate for each group.

    Returns:
      A tuple containing y_0, z_0, y_1, z_1
    """

    z_0 = np.random.uniform(0, 1, n_samples)
    z_1 = np.random.uniform(0, 1, n_samples)
    y_0 = f0(z_0) + np.random.normal(0, 1, n_samples)
    y_1 = f1(z_1) + np.random.normal(0, 1, n_samples)


    return y_0, z_0, y_1, z_1

def new_generate_data_normal(f0, f1, n_samples):
    """
    Y(0) = f0(Z(0)) + N(0, I), Z(0) ~ U[0,1],
    Y(1) = f1(Z(1)) + N(0, I), Z(1) ~ U[0,1].
    Generates synthetic data for the control and treatment groups.

    Args:
      n_samples: The number of samples to generate for each group.

    Returns:
      A tuple containing y_0, z_0, y_1, z_1
    """

    z_0 = np.random.normal(0, 1, n_samples)
    z_1 = np.random.normal(0, 1, n_samples)
    y_0 = f0(z_0) + np.random.normal(0, 1, n_samples)
    y_1 = f1(z_1) + np.random.normal(0, 1, n_samples)


    return y_0, z_0, y_1, z_1


def cluster_z_values(data, c=0.3):
    """
    Divides the interval [0, 1] into subintervals and clusters z-values.

    Args:
        data = (y,z): A NumPy array of shape (n_samples, 2) where the second column represents z-values.

    Returns:
        A NumPy array of the same shape as data, but with clustered z-values.
    """

    # Calculate the number of subintervals
    n_samples = data.shape[0]
    num_subintervals = int(c * n_samples**(1/3))  # we can choose a constant C here -> TUNING
    if num_subintervals == 0:
        num_subintervals = 1

    # Define the boundaries of subintervals
    interval_width = 1.0 / num_subintervals
    boundaries = np.linspace(min(data[:, 1]), max(data[:, 1]), num_subintervals + 1)

    # Cluster z-values to the subintervals
    clustered_z = np.zeros(n_samples)
    for i in range(n_samples):
      z = data[i, 1]
      for j in range(num_subintervals):
        if boundaries[j] <= z < boundaries[j+1]:
          clustered_z[i] = boundaries[j] + interval_width / 2
          break
      #handle edge case where z == 1
      if clustered_z[i] == 0 and z == 1:
          clustered_z[i] = boundaries[-2] + interval_width/2

    # Return data with clustered z values
    clustered_data = np.column_stack((data[:, 0], clustered_z))

    return clustered_data


from collections import defaultdict

def empirical_distribution(data):
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
    weights = counts / len(data)
    return support, weights


def calculate_empirical_distributions(clustered_data):
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

    z_support, z_weights = empirical_distribution(z_values)

    conditional_y_distributions = defaultdict(lambda: (np.array([]), np.array([])))

    for z in z_support:
        indices = np.where(z_values == z)
        corresponding_y = y_values[indices]
        y_support, y_weights = empirical_distribution(corresponding_y)
        conditional_y_distributions[z] = (y_support, y_weights)

    return z_support, z_weights, conditional_y_distributions


from ot import emd

def ot_compute(dist_a, dist_b, p):
    """Computes the optimal transport (OT) distance between two empirical distributions.

    Args:
        dist_a: A tuple containing the support and weights of the first distribution.
        dist_b: A tuple containing the support and weights of the second distribution.

    Returns:
        A tuple containing the optimal transport distance and the transport matrix.
    """
    support_a, weights_a = dist_a
    support_b, weights_b = dist_b

    # Compute the cost matrix
    M = np.abs(np.subtract.outer(support_a, support_b)) ** p


    # Compute the OT distance and the transport matrix
    transport_matrix = emd(weights_a, weights_b, M)
    ot_distance = np.sum(M * transport_matrix)

    return ot_distance, transport_matrix
    

def cot_estimator(control_data, treatment_data, c=1.):
    """
    Estimates the Conditional Optimal Transport (COT) distance between two datasets.
    """

    clustered_control_data = cluster_z_values(control_data, c)
    clustered_treatment_data = cluster_z_values(treatment_data, c)

    # Step 1: Compute empirical distributions for each dataset
    control_z_support, control_z_weights, control_conditional_y = calculate_empirical_distributions(clustered_control_data)
    treatment_z_support, treatment_z_weights, treatment_conditional_y = calculate_empirical_distributions(clustered_treatment_data)

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



def MirrorOT(control_data, treatment_data, eta):
    """
    Computes the optimal transport matrix and costs for different eta values.
    """

    M1 = np.abs(np.subtract.outer(control_data[:,0], treatment_data[:,0])) ** 2 \
         + eta * np.abs(np.subtract.outer(control_data[:,1], treatment_data[:,1]))

    M2 = np.abs(np.subtract.outer(control_data[:,0], treatment_data[:,0])) ** 2

    control_size = control_data.shape[0]
    treatment_size = treatment_data.shape[0]

    # Compute the optimal transport matrix
    L = emd(np.ones(control_size) / control_size, np.ones(treatment_size) / treatment_size, M1)
    cost_m2 = np.sum(M2 * L)

    return L, M1, M2, cost_m2



def plot_compare_COT_mirror(x, y, c, truev=None, ll=None):
    import matplotlib.pyplot as plt
    
    # Create the plot
    plt.plot(x, y)
    
    # Plot the horizontal line at y = c
    plt.axhline(y=c, color='b', linestyle='--', label='COT')
    if truev is not None:
        plt.axhline(y=truev, color='r', linestyle='--', label='True')
    if ll is not None:
        for j, v in enumerate(ll):
            plt.axhline(y=v, linestyle='--', label=f'value-{j}')
    
    
    # Adding labels and title
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.title('Plot of Curve with Horizontal Line')
    
    # Show legend
    plt.legend()
    
    # Display the plot
    plt.show()








