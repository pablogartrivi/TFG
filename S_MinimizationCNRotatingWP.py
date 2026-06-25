import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D  # noqa
from Mueller import Mueller
from RotatingWPPolCN import RotatingWPPolCN
from RotatingWSPPolSops import RotatingWPPolSops

# --------------------------------------------------
# Parámetros iniciales
# --------------------------------------------------

Sini = np.array([1, 1, 0, 0])
philamina = np.pi / 2

# --------------------------------------------------
# Esfera de Poincaré
# --------------------------------------------------

fact = 0.96
u = np.linspace(0, np.pi, 60)
v = np.linspace(0, 2*np.pi, 60)

X = fact * np.outer(np.sin(u), np.cos(v))
Y = fact * np.outer(np.sin(u), np.sin(v))
Z = fact * np.outer(np.cos(u), np.ones_like(v))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, color='lightgray', alpha=0.5)

ax.set_xlabel('S1')
ax.set_ylabel('S2')
ax.set_zlabel('S3')

# --------------------------------------------------
# Estados de polarización (figura en 8)
# --------------------------------------------------

Nrot = 100
rot = np.linspace(0, np.pi, Nrot)

SoPs = np.zeros((Nrot, 4))

for k in range(Nrot):
    M = Mueller('waveplate', rot[k], philamina)
    SoPs[k, :] = M @ Sini

ax.scatter(SoPs[:, 1], SoPs[:, 2], SoPs[:, 3],
           color='blue', s=15)

# --------------------------------------------------
# Función objetivo (CN)
# --------------------------------------------------

def objective(x):
    return RotatingWPPolCN(x)

# --------------------------------------------------
# Optimización inicial
# --------------------------------------------------

Nrot_opt = 6

x0 = np.pi * np.random.rand(Nrot_opt)

bounds = [(0, np.pi)] * Nrot_opt

res = minimize(
    objective,
    x0,
    method='L-BFGS-B',
    bounds=bounds,
    options={'maxiter': 300}
)

x_best = res.x

CN_best, TSoP = RotatingWPPolSops(x_best, philamina)

# --------------------------------------------------
# Guardar mejor solución inicial
# --------------------------------------------------

CNB = CN_best
XB = x_best.copy()

# --------------------------------------------------
# Plot solución inicial
# --------------------------------------------------

ax.scatter(TSoP[:, 1], TSoP[:, 2], TSoP[:, 3],
           color='green', s=40)

# --------------------------------------------------
# Búsqueda global (equivalente al bucle MATLAB)
# --------------------------------------------------

nitera = 10

for n in range(nitera):

    x0 = np.pi * np.random.rand(Nrot_opt)

    res = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 300}
    )

    x = res.x
    CN, TSoP = RotatingWPPolSops(x, philamina)

    if CN < CNB:
        CNB = CN
        XB = x.copy()

    print(f"Iter {n+1} | CN = {CN:.6e} | Best CN = {CNB:.6e}")

# --------------------------------------------------
# Plot mejor solución
# --------------------------------------------------

ax.scatter(TSoP[:, 1], TSoP[:, 2], TSoP[:, 3],
           color='red', s=40)

plt.title("Poincaré Sphere - Optimization of CN")
plt.show()

# --------------------------------------------------
# Ángulos óptimos finales
# --------------------------------------------------

rotacion = np.sort(XB * 180 / np.pi)

print("\nOptimal angles (deg):")
print(rotacion)