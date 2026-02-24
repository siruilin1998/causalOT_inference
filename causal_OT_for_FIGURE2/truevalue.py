import numpy as np
import numpy.typing as npt

def true_vip1dim(
        eta: float, 
        b0: float,
        b1: float, 
        var0: float = 1.0,
        var1: float = 1.0
    ) -> float:
    """
    Compute the true value of Vip(eta) for Gaussian regression model with 1dim outcome and 1dim covariate.

    Parameters
    ----------
    eta : regularization parameter in E[(y0+y1)**2] + eta * E[(z0-z1)**2].
    b0 : the coefficient from y0 = b0 z + e0, z~N(0,1), e0~N(0,var0).
    b1 : the coefficient from y1 = b1 z + e1, z~N(0,1), e1~N(0,var1).
    var0 : optional. The default is 1.0.
    var1 : optional. The default is 1.0.

    Returns
    -------
    result: true Vip(eta) value.

    """
    s0 = var0 ** 0.5
    s1 = var1 ** 0.5
    
    numerator = (b0**2 + s0**2) * (b1**2 + s1**2) - eta * b0 * b1 + eta * s0 * s1
    denominator = ((b0**2 + s0**2) * (b1**2 + s1**2) - 2 * eta * b0 * b1 + eta**2 + 2 * eta * s0 * s1)**0.5
    result = (b0**2 + b1**2 + s0**2 + s1**2) - 2 * numerator / denominator
    return result


def true_vipmdim(
        eta: float,
        b0: npt.NDArray[np.float64],
        b1: npt.NDArray[np.float64],
        S0: npt.NDArray[np.float64] = None,
        S1: npt.NDArray[np.float64] = None,
        scale_factor: float = 1.
    ):
    """
    Compute the true value of Vip(eta) for Gaussian regression model with multi-dim outcome and multi-dim covariate.

    Parameters
    ----------
    eta : regularization parameter in E[norm(y0+y1)**2] + eta * E[norm(z0-z1)**2].
    b0 : the coefficient from y0 = b0 z + e0, z~N(0,Ip), e0~N(0,S1).
    b1 : the coefficient from y1 = b1 z + e1, z~N(0,Ip), e1~N(0,S2).
    S0 : optional. The default is identity.
    S1 : optional. The default is identity.


    Returns
    -------
    result: true Vip(eta) value.

    """
    eta += 1e-8
    
    if b0.shape != b1.shape:
        raise ValueError(f'The shape of b0 {b0.shape} does not match that of b1 {b1.shape}.')
    
    if S0 is not None:
        if not np.array_equal(S0, S0.T):
            raise ValueError('S0 needs to be symmetric matrix.')
        if b0.shape[0] != S0.shape[0]:
            raise ValueError(f'The first dim of b0:{b0.shape[0]} does not match the first dim of S0:{S0.shape[0]}.')
    else:
        S0 = np.eye(b0.shape[0])
    
    if S1 is not None:
        if not np.array_equal(S1, S1.T):
            raise ValueError('S1 needs to be symmetric matrix.')
        if b1.shape[0] != S1.shape[0]:
            raise ValueError(f'The first dim of b1:{b1.shape[0]} does not match the first dim of S1:{S1.shape[0]}.')
    else:
        S1 = np.eye(b1.shape[0])
    
    
    # Define Sigma_0
    Sigma_0 = np.block([
        [scale_factor ** (-1) * (b0 @ b0.T + S0), np.sqrt(eta) * scale_factor ** (-1) * b0],
        [np.sqrt(eta) * scale_factor ** (-1) * b0.T, scale_factor ** (-1) * eta * np.eye(b0.shape[1])]
    ])
    
    # Define Sigma_1
    Sigma_1 = np.block([
        [scale_factor ** (-1) * (b1 @ b1.T + S1), - scale_factor ** (-1) * np.sqrt(eta) * b1],
        [- scale_factor ** (-1) * np.sqrt(eta) * b1.T, scale_factor ** (-1) * eta * np.eye(b1.shape[1])]
    ])
    
    e = np.vstack((np.eye(b0.shape[0]), np.zeros((b0.shape[1], b0.shape[0]))))
    
    eigvals_0, eigvecs_0 = np.linalg.eigh(Sigma_0)
    
    Sigma_0_half = eigvecs_0 @ np.diag(np.sqrt(eigvals_0)) @ eigvecs_0.T  # Sigma_0^(1/2)
    Sigma_0_inv_half = eigvecs_0 @ np.diag(1 / np.sqrt(eigvals_0)) @ eigvecs_0.T  # Sigma_0^(-1/2)
    
    # Step 2: Compute intermediate matrix
    intermediate = Sigma_0_half @ Sigma_1 @ Sigma_0_half
    
    # Step 3: Square root of the intermediate matrix
    eigvals_inter, eigvecs_inter = np.linalg.eigh(intermediate)
    intermediate_half = eigvecs_inter @ np.diag(np.sqrt(eigvals_inter)) @ eigvecs_inter.T  # (Sigma_0^(1/2)Sigma_1Sigma_0^(1/2))^(1/2)
    
    # Step 4: Compute the full expression
    result_matrix = Sigma_0_half @ intermediate_half @ Sigma_0_inv_half @ e @ e.T
    trace_result = np.trace(result_matrix)
    
    q = trace_result
    
    result = np.trace(b0 @ b0.T + S0) + np.trace(b1 @ b1.T + S1) - 2 * scale_factor * q
   
    return result



def true_vc1dim(
        b0: float,
        b1: float, 
        var0: float = 1.0,
        var1: float = 1.0
    ) -> float:
    '''
    Compute the true Vc value for Gaussian regression model with 1dim outcome and 1dim covariate.

    Parameters
    ----------
    b0 : the coefficient from y0 = b0 z + e0, z~N(0,1), e0~N(0,var0).
    b1 : the coefficient from y1 = b1 z + e1, z~N(0,1), e1~N(0,var1).
    var0 : optional. The default is 1.0.
    var1 : optional. The default is 1.0.

    Returns
    -------
    result: true Vc value.

    '''
    
    result = (b0 + b1) ** 2 + (var0 ** 0.5 - var1 ** 0.5) ** 2
    return result


def _half(S):
    #return the square root of matrix S
    eigvals, eigvecs = np.linalg.eigh(S)
    S_half = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    
    return S_half

def true_vcmdim(
        b0: npt.NDArray[np.float64],
        b1: npt.NDArray[np.float64],
        S0: npt.NDArray[np.float64] = None,
        S1: npt.NDArray[np.float64] = None,
        a: npt.NDArray[np.float64] = None,
    ) -> float:
    """
    Compute the true value of Vc for Gaussian regression model with multi-dim outcome and multi-dim covariate.

    Parameters
    ----------
    b0 : the coefficient from y0 = b0 z + e0, z~N(0,Ip), e0~N(0,S1).
    b1 : the coefficient from y1 = b1 z + e1, z~N(0,Ip), e1~N(0,S2).
    S0 : optional. The default is identity.
    S1 : optional. The default is identity.


    Returns
    -------
    result: true Vc value.

    """
    
    
    if b0.shape != b1.shape:
        raise ValueError(f'The shape of b0 {b0.shape} does not match that of b1 {b1.shape}.')
    
    if S0 is not None:
        if not np.array_equal(S0, S0.T):
            raise ValueError('S0 needs to be symmetric matrix.')
        if b0.shape[0] != S0.shape[0]:
            raise ValueError(f'The first dim of b0:{b0.shape[0]} does not match the first dim of S0:{S0.shape[0]}.')
    else:
        if a is not None:
            S0 = np.eye(b0.shape[0]) * a
        else:
            S0 = np.eye(b0.shape[0])
    
    if S1 is not None:
        if not np.array_equal(S1, S1.T):
            raise ValueError('S1 needs to be symmetric matrix.')
        if b1.shape[0] != S1.shape[0]:
            raise ValueError(f'The first dim of b1:{b1.shape[0]} does not match the first dim of S1:{S1.shape[0]}.')
    else:
        if a is not None:
            S1 = np.eye(b1.shape[0]) * a
        else:
            S1 = np.eye(b1.shape[0])
        
    Ez = np.trace((b0 + b1) @ (b0 + b1).T)
    
    intermediate = _half(S0) @ S1 @ _half(S0)
    
    trace_term = np.trace(S0 + S1 - 2 * _half(intermediate))
    
    return Ez + trace_term
    

def true_vc_square(
        b0: npt.NDArray[np.float64],
        b1: npt.NDArray[np.float64],
        S0: npt.NDArray[np.float64] = None,
        S1: npt.NDArray[np.float64] = None
    ) -> float:
    """
    Compute the true value of Vc for following regression model with multi-dim outcome and multi-dim covariate.

    Parameters
    ----------
    b0 : the coefficient from y0 = b0 z ** 2 + e0, z~N(0,Ip), e0~N(0,S1).
    b1 : the coefficient from y1 = b1 z ** 2 + e1, z~N(0,Ip), e1~N(0,S2).
    S0 : optional. The default is identity.
    S1 : optional. The default is identity.


    Returns
    -------
    result: true Vc value.

    """
    
    
    if b0.shape != b1.shape:
        raise ValueError(f'The shape of b0 {b0.shape} does not match that of b1 {b1.shape}.')
    
    if S0 is not None:
        if not np.array_equal(S0, S0.T):
            raise ValueError('S0 needs to be symmetric matrix.')
        if b0.shape[0] != S0.shape[0]:
            raise ValueError(f'The first dim of b0:{b0.shape[0]} does not match the first dim of S0:{S0.shape[0]}.')
    else:
        S0 = np.eye(b0.shape[0])
    
    if S1 is not None:
        if not np.array_equal(S1, S1.T):
            raise ValueError('S1 needs to be symmetric matrix.')
        if b1.shape[0] != S1.shape[0]:
            raise ValueError(f'The first dim of b1:{b1.shape[0]} does not match the first dim of S1:{S1.shape[0]}.')
    else:
        S1 = np.eye(b1.shape[0])
        
        
    F = np.ones((b0.shape[1], b0.shape[1]))
    F += 2 * np.eye(b0.shape[1])
    
    Ez = np.trace((b0 + b1) @ F @ (b0 + b1).T)
    
    intermediate = _half(S0) @ S1 @ _half(S0)
    
    trace_term = np.trace(S0 + S1 - 2 * _half(intermediate))
    
    return Ez + trace_term   

def true_vc_prod1dim(
        b0: float,
        b1: float, 
        k0: float,
        k1: float,
        SIM: int = 5000,
    ) -> float:
    '''
    Compute the true Vc value for Gaussian heteroskedastic regression model with 1dim outcome and 1dim covariate.
    
    Parameters
    ----------
    b0, k0 : the coefficient from y0 = (b0 z + k0) * e0, z~N(0,1), e0~N(0,1).
    b1, k1 : the coefficient from y1 = (b1 z + k1) * e1, z~N(0,1), e0~N(0,1).
    SIM: Monte Carlo repetitions.
    
    Returns
    -------
    result: MC-based approximation of true Vc value.
    
    ''' 
    Z = np.random.randn(SIM)

    # Compute the two terms inside the absolute value
    term1 = np.abs(b0 * Z + k0)
    term2 = np.abs(b1 * Z + k1)

    # Compute the squared difference
    squared_diff = (term1 - term2) ** 2

    # Approximate the expectation by taking the mean
    result = np.mean(squared_diff)

    return result

    
    

def true_vu1dim(
        b0: float,
        b1: float, 
        var0: float = 1.0,
        var1: float = 1.0
    ) -> float:
    '''
    Compute the true Vu value for Gaussian regression model with 1dim outcome and 1dim covariate.

    Parameters
    ----------
    b0 : the coefficient from y0 = b0 z + e0, z~N(0,1), e0~N(0,var0).
    b1 : the coefficient from y1 = b1 z + e1, z~N(0,1), e1~N(0,var1).
    var0 : optional. The default is 1.0.
    var1 : optional. The default is 1.0.

    Returns
    -------
    result: true Vu value.

    '''
    
    result = ((b0 ** 2 + var0) ** 0.5 - (b1 ** 2 + var1) ** 0.5) ** 2
    return result
    
def true_vumdim(
        b0: npt.NDArray[np.float64],
        b1: npt.NDArray[np.float64],
        S0: npt.NDArray[np.float64] = None,
        S1: npt.NDArray[np.float64] = None,
        a: npt.NDArray[np.float64] = None,
    ) -> float:
    '''
    Compute the true Vu value for Gaussian regression model with multi-dim outcome and multi-dim covariate.

    Parameters
    ----------
    b0 : the coefficient from y0 = b0 z + e0, z~N(0,Ip), e0~N(0,S1).
    b1 : the coefficient from y1 = b1 z + e1, z~N(0,Ip), e1~N(0,S2).
    S0 : optional. The default is identity.
    S1 : optional. The default is identity.

    Returns
    -------
    result : true Vu value

    '''
    
    if b0.shape != b1.shape:
        raise ValueError(f'The shape of b0 {b0.shape} does not match that of b1 {b1.shape}.')
    
    if S0 is not None:
        if not np.array_equal(S0, S0.T):
            raise ValueError('S0 needs to be symmetric matrix.')
        if b0.shape[0] != S0.shape[0]:
            raise ValueError(f'The first dim of b0:{b0.shape[0]} does not match the first dim of S0:{S0.shape[0]}.')
    else:
        if a is not None:
            S0 = np.eye(b0.shape[0]) * a
        else:
            S0 = np.eye(b0.shape[0])
    
    if S1 is not None:
        if not np.array_equal(S1, S1.T):
            raise ValueError('S1 needs to be symmetric matrix.')
        if b1.shape[0] != S1.shape[0]:
            raise ValueError(f'The first dim of b1:{b1.shape[0]} does not match the first dim of S1:{S1.shape[0]}.')
    else:
        if a is not None:
            S1 = np.eye(b1.shape[0]) * a
        else:
            S1 = np.eye(b1.shape[0])
        
    Sigma0 = b0 @ b0.T + S0
    Sigma1 = b1 @ b1.T + S1
    
    intermediate = _half(Sigma0) @ Sigma1 @ _half(Sigma0)
    
    result = np.trace(Sigma0 + Sigma1 - 2 * _half(intermediate))
    
    return result


def true_vc_multi_1dim(
        b0: float,
        b1: float, 
        var0: float = 1.0,
        var1: float = 1.0
    ) -> float:
    '''
    Compute the true Vc value for Gaussian regression model with 1dim outcome and 1dim covariate.

    Parameters
    ----------
    b0 : the coefficient from y0 = b0 z odot e0, z~N(0,1), e0~N(0,var0).
    b1 : the coefficient from y1 = b1 z odot e1, z~N(0,1), e1~N(0,var1).
    var0 : optional. The default is 1.0.
    var1 : optional. The default is 1.0.

    Returns
    -------
    result: true Vc value.

    '''
    
    var0 *= b0 ** 2
    var1 *= b1 ** 2
    
    result = (var0 ** 0.5 - var1 ** 0.5) ** 2
    return result


