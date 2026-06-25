import numpy as np
from Mueller import Mueller

def TurnMatrixM(M, Rotation):
    """
    Rotates a Mueller matrix using Stokes-space rotation.

    Parameters
    ----------
    M : np.ndarray (4x4)
        Mueller matrix of the optical element
    Rotation : float
        Rotation angle (radians)

    Returns
    -------
    TM : np.ndarray (4x4)
        Rotated Mueller matrix
    """

    R_left = Mueller('rotated', -Rotation, 0.0)
    R_right = Mueller('rotated', Rotation, 0.0)

    TM = R_left @ M @ R_right

    return TM