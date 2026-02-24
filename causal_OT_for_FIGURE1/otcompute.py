import ot
import numpy as np
import numpy.typing as npt
from typing import Tuple

from costmat import cost_mat_root_y1p2square, cost_matrix

def ot_compute(
        n1: int, 
        n2: int, 
        cmat1: npt.NDArray[np.float64], 
        cmat2: npt.NDArray[np.float64]
    ) -> Tuple[float, float]:
    """
    Compute the OT distance between two uniform distributions using the given two cost matrix.    

    Parameters
    ----------
    n1 : the size of source distribution.
    n2 : the size of target distribution.
    cmat1 : A NumPy array of shape (n1, n2), the matrix used to compute OT map.
    cmat2 : A NumPy array of shape (n1, n2), the matrix used to compute transport value.

    Returns
    -------
    v1 : the transport cost with cost_mat.
    v2 : the transport cost with cost_mat2.

    """
    if cmat1.shape != (n1, n2):
        raise ValueError(f'The shape of cost matrix1: {cmat1.shape} does not match (n1,n2): {(n1, n2)}.')
    if cmat2.shape != (n1, n2):
        raise ValueError(f'The shape of cost matrix2: {cmat2.shape} does not match (n1,n2): {(n1, n2)}.')

    
    a = np.ones(n1) / n1  # Uniform distribution for source
    b = np.ones(n2) / n2  # Uniform distribution for target
      
    # Compute the optimal transport distance using ot.emd()
    G = ot.emd(a, b, cmat1, numItermax=500000)
    # G = ot.emd(a, b, cmat1)
    
    v1 = np.sum(G * cmat1)
    v2 = np.sum(G * cmat2)
    return v1, v2


def vip_estimate(
        A1 : npt.NDArray[np.float64], 
        A2 : npt.NDArray[np.float64], 
        dy : int, 
        dz : int, 
        etalist : list[float]
    ) -> list[float]:
    '''
    Compute the Vip estimate given data A1 (Y0, Z), A2 (Y1, Z), and etalist.

    Parameters
    ----------
    A1 : npt.NDArray[np.float64]
        A Numpy array of (Y0, Z) samples.
    A2 : npt.NDArray[np.float64]
        A Numpy array of (Y1, Z) samples.
    dy : int
        Outcome (y) dimension.
    dz : int
        Covariate (z) dimension.
    etalist : list[float]
        List of regularization para eta.

    Returns
    -------
    distance_noetalist : list[float]
        List of estimates for different eta's.

    '''
    
    M1, M2 = cost_mat_root_y1p2square(A1, A2, dy, dz)
    cost_mat2 = M1

    distance_noetalist = []

    for eta in etalist:
        cost_mat = cost_matrix(M1, M2, eta)
        _, distance_noeta = ot_compute(A1.shape[0], A2.shape[0], cost_mat, cost_mat2)
        distance_noetalist.append(distance_noeta)
    
    return distance_noetalist


