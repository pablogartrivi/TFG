import numpy as np
from Mueller import Mueller

def RotatingWPPolCN(rot):
    """
    Computes the condition number of a rotating waveplate polarimeter.

    Parameters
    ----------
    rot : array-like
        Rotation angles (radians)

    Returns
    -------
    CN : float
        Condition number of the system matrix A
    """

    philamina = np.pi / 2  # quarter-wave plate
    Sini = np.array([1, 1, 0, 0])

    rot = np.array(rot)
    Nrot = len(rot)

    A = np.zeros((Nrot, 4))

    for k in range(Nrot):
        M = Mueller('waveplate', rot[k], philamina)
        A[k, :] = M @ Sini

    CN = np.linalg.cond(A)

    return CN