import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# ÁNGULOS
# ============================================================

angulos_deg = np.array([32, 52, 80, 101, 128, 148])

# ============================================================
# PARÁMETROS
# ============================================================

Sini = np.array([1, 1, 0, 0])

# Quarter-wave plate
philamina = np.pi / 2

# ============================================================
# MATRIZ DE MUELLER
# ============================================================

def MuellerWaveplate(theta, delta):

    c = np.cos(2 * theta)
    s = np.sin(2 * theta)

    M = np.array([

        [1, 0, 0, 0],

        [0,
         c**2 + s**2*np.cos(delta),
         c*s*(1 - np.cos(delta)),
         -s*np.sin(delta)],

        [0,
         c*s*(1 - np.cos(delta)),
         s**2 + c**2*np.cos(delta),
         c*np.sin(delta)],

        [0,
         s*np.sin(delta),
         -c*np.sin(delta),
         np.cos(delta)]

    ])

    return M

# ============================================================
# GENERAR LOS ESTADOS INTRODUCIDOS
# ============================================================

N = len(angulos_deg)

SoPs = np.zeros((N, 4))

for k in range(N):

    theta = np.radians(angulos_deg[k])

    M = MuellerWaveplate(theta, philamina)

    SoPs[k, :] = M @ Sini

# ============================================================
# MATRIZ INSTRUMENTAL
# ============================================================

A = SoPs

# ============================================================
# CONDITION NUMBER
# ============================================================

CN = np.linalg.cond(A)

print("====================================")
print(" CONDITION NUMBER")
print("====================================")
print(f"CN = {CN:.6f}")

# ============================================================
# GENERAR LA CURVA EN "8"
# ============================================================

Nrot = 400
rot = np.linspace(0, np.pi, Nrot)

SoPs_curve = np.zeros((Nrot, 4))

for k in range(Nrot):

    M = MuellerWaveplate(rot[k], philamina)

    SoPs_curve[k, :] = M @ Sini

# ============================================================
# ESFERA DE POINCARÉ
# ============================================================

fact = 0.96

u = np.linspace(0, np.pi, 80)
v = np.linspace(0, 2*np.pi, 80)

X = fact * np.outer(np.sin(u), np.cos(v))
Y = fact * np.outer(np.sin(u), np.sin(v))
Z = fact * np.outer(np.cos(u), np.ones_like(v))

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

# ============================================================
# SUPERFICIE ESFERA
# ============================================================

ax.plot_surface(
    X, Y, Z,
    color='lightgray',
    alpha=0.15,
    linewidth=0
)

# ============================================================
# CURVA EN "8"
# ============================================================

ax.plot(
    SoPs_curve[:,1],
    SoPs_curve[:,2],
    SoPs_curve[:,3],
    linewidth=2,
    color='blue'
)

# ============================================================
# PUNTOS SELECCIONADOS
# ============================================================

ax.scatter(
    SoPs[:,1],
    SoPs[:,2],
    SoPs[:,3],
    color='red',
    s=90
)

# ============================================================
# ETIQUETAS
# ============================================================

for i in range(N):

    ax.text(
        SoPs[i,1],
        SoPs[i,2],
        SoPs[i,3],
        f'{i+1}',
        fontsize=11
    )

# ============================================================
# EJES
# ============================================================

ax.plot([-1,1],[0,0],[0,0], linewidth=1.5)
ax.plot([0,0],[-1,1],[0,0], linewidth=1.5)
ax.plot([0,0],[0,0],[-1,1], linewidth=1.5)

ax.text(1.15,0,0,'S1',fontsize=15)
ax.text(0,1.15,0,'S2',fontsize=15)
ax.text(0,0,1.15,'S3',fontsize=15)

# ============================================================
# CONFIGURACIÓN VISUAL
# ============================================================

ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])

ax.set_box_aspect([1,1,1])

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.grid(False)

plt.title(
    f'Poincaré Sphere\nCondition Number = {CN:.4f}',
    fontsize=16
)

plt.tight_layout()
plt.show()

# ============================================================
# MOSTRAR ÁNGULOS
# ============================================================

print("\nÁngulos utilizados (deg):")
print(np.sort(angulos_deg))