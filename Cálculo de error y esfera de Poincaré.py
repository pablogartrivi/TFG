import numpy as np
import matplotlib.pyplot as plt


def normalizar_stokes(S):
    """
    Convierte una matriz de Stokes Nx4 a coordenadas de Poincaré Nx3:
    (s1, s2, s3) = (S1/S0, S2/S0, S3/S0)
    """
    S = np.asarray(S, dtype=np.float64)

    if S.ndim == 1:
        S = S.reshape(1, -1)

    if S.shape[1] != 4:
        raise ValueError("S debe tener forma (N, 4)")

    S0 = S[:, 0]
    if np.any(np.abs(S0) < 1e-12):
        raise ValueError("Algún S0 es demasiado pequeño o cero")

    s1 = S[:, 1] / S0
    s2 = S[:, 2] / S0
    s3 = S[:, 3] / S0

    return np.column_stack([s1, s2, s3])


def plot_esfera_poincare(S_exp, S_theo, labels=None, title="Esfera de Poincaré"):
    """
    Dibuja en la esfera de Poincaré los puntos experimentales y teóricos.
    """
    P_exp = normalizar_stokes(S_exp)
    P_theo = normalizar_stokes(S_theo)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Esfera
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 40)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(x, y, z, color='lightgray', alpha=0.15, linewidth=0)
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.25, linewidth=0.5)

    # Ejes
    ax.plot([-1, 1], [0, 0], [0, 0], color='black', linewidth=1)
    ax.plot([0, 0], [-1, 1], [0, 0], color='black', linewidth=1)
    ax.plot([0, 0], [0, 0], [-1, 1], color='black', linewidth=1)

    # Puntos experimentales
    ax.scatter(
        P_exp[:, 0], P_exp[:, 1], P_exp[:, 2],
        c='tab:blue', s=60, label='Experimental', depthshade=True
    )

    # Puntos teóricos
    ax.scatter(
        P_theo[:, 0], P_theo[:, 1], P_theo[:, 2],
        c='tab:red', s=60, marker='^', label='Teórico', depthshade=True
    )

    # Líneas que unen exp y teórico
    n = min(len(P_exp), len(P_theo))
    for i in range(n):
        ax.plot(
            [P_exp[i, 0], P_theo[i, 0]],
            [P_exp[i, 1], P_theo[i, 1]],
            [P_exp[i, 2], P_theo[i, 2]],
            color='gray', alpha=0.5, linewidth=1
        )

    # Etiquetas
    if labels is not None:
        for i, lab in enumerate(labels):
            ax.text(P_exp[i, 0], P_exp[i, 1], P_exp[i, 2], f"  {lab}", fontsize=9)

    ax.set_xlabel("s1")
    ax.set_ylabel("s2")
    ax.set_zlabel("s3")
    ax.set_title(title)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_errores(err_pct, err_rel, labels=None):
    """
    Grafica el porcentaje de error y el error relativo.
    """
    err_pct = np.asarray(err_pct, dtype=np.float64).ravel()
    err_rel = np.asarray(err_rel, dtype=np.float64).ravel()

    if len(err_pct) != len(err_rel):
        raise ValueError("err_pct y err_rel deben tener la misma longitud")

    x = np.arange(len(err_pct))
    if labels is None:
        labels = [str(i) for i in x]

    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)

    axs[0].plot(x, err_pct, marker='o')
    axs[0].set_ylabel("% error")
    axs[0].set_title("Porcentaje de error")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(x, err_rel, marker='o')
    axs[1].set_ylabel("Error relativo")
    axs[1].set_xlabel("Medida")
    axs[1].set_title("Error relativo")
    axs[1].grid(True, alpha=0.3)

    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels, rotation=45, ha='right')

    plt.show()


# =========================================================
# EJEMPLO DE USO
# =========================================================
if __name__ == "__main__":

    # Matriz de Stokes experimentales
    S_exp = np.array([

    ], dtype=np.float64)

    # Matriz de Stokes teóricos


    S_theo = np.array([

    ], dtype=np.float64)



    err_pct = np.array([

    ], dtype=np.float64)

    err_rel = np.array([
    ], dtype=np.float64)

    labels = [
    "M1", "M2", "M3", "M4", "M5",
    "M6", "M7", "M8", "M9", "M10",
    "M11", "M12", "M13", "M14", "M15",
    "M16", "M17", "M18", "M19"
    ]

    plot_esfera_poincare(S_exp, S_theo, labels=labels)
    plot_errores(err_pct, err_rel, labels=labels)