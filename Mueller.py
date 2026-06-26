import numpy as np

def Mueller(Type, Rotation, Phase):
    """
    Generate the Mueller matrix of an optical element.

    Parameters
    ----------
    Type : str
        'polarizer', 'waveplate', or 'rotated'
    Rotation : float
        Rotation angle (radians)
    Phase : float
        Retardance (radians), used for waveplates

    Returns
    -------
    M : np.ndarray (4x4)
        Mueller matrix
    """

    def TurnMatrixM(M, theta):
        R = Mueller('rotated', -theta, 0.0)
        R_inv = Mueller('rotated', theta, 0.0)
        return R @ M @ R_inv

    M = np.zeros((4, 4))

    t = Type.lower()

    if t in ['polarizer', 'polarizador']:
        M = np.zeros((4, 4))
        M[0:2, 0:2] = 0.5
        M = TurnMatrixM(M, Rotation)

    elif t in ['waveplate', 'retardador']:
        M = np.eye(4)
        M[2, 2] = np.cos(Phase)
        M[3, 3] = np.cos(Phase)
        M[2, 3] = np.sin(Phase)
        M[3, 2] = -np.sin(Phase)
        M = TurnMatrixM(M, Rotation)

    elif t == 'rotated':
        M = np.eye(4)
        M[1, 1] = np.cos(2 * Rotation)
        M[2, 2] = np.cos(2 * Rotation)
        M[1, 2] = np.sin(2 * Rotation)
        M[2, 1] = -np.sin(2 * Rotation)

    return M