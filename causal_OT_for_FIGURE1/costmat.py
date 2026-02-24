import numpy as np
import numpy.typing as npt
from typing import Tuple

def cost_mat_root_y1p2square(
        A1: npt.NDArray[np.float64], 
        A2: npt.NDArray[np.float64], 
        dy: int = 1, 
        dz: int = 1
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute the OT cost matrix between (y1, y2) and (z1, z2).
    
    Parameters
    ----------
    A1: A NumPy array of shape (n1, dy + dz) where each row represents (y1, z1).
    A2: A NumPy array of shape (n2, dy + dz) where each row represents (y2, z2).

    Returns
    -------
    mat1 : A NumPy array of shape (n1, n2)
        norm(y1 + y2)**2
    mat2 : A NumPy array of shape (n1, n2)
        norm(z1 - z2)**2.
    """
    if A1.shape[1] != dy + dz:
        raise ValueError(f'The second dimension of A1: {A1.shape[1]} does not match dy + dz: {dy + dz}.')
    if A2.shape[1] != dy + dz:
        raise ValueError(f'The second dimension of A2: {A2.shape[1]} does not match dy + dz: {dy + dz}.')    
    
    # Extract y1, z1 from A1 and y2, z2 from A2
    y1 = A1[:, :dy]
    y2 = A2[:, :dy]
    
    # Compute M1  
    mat1 = np.sum((y1[:, np.newaxis, :] + y2[np.newaxis, :, :])**2, axis=2)

    z1 = A1[:, dy:]
    z2 = A2[:, dy:]
    
    # Compute M2  
    mat2 = np.sum((z1[:, np.newaxis, :] - z2[np.newaxis, :, :])**2, axis=2)
    
    return mat1, mat2

def cost_matrix(M1, M2, eta=1):
    if M1.shape != M2.shape:
        raise ValueError(f'The shape of M1:{M1.shape} does not match that of M2:{M2.shape}.')
    return M1 + eta * M2