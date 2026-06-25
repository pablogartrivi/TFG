import os
import rawpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def guardar_roi_debug(img, y1, y2, x1, x2, out_path, title="ROI"):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    rect = patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        linewidth=2,
        edgecolor='r',
        facecolor='none'
    )
    ax.add_patch(rect)
    ax.set_title(title)
    ax.axis("off")
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()


def leer_rojo_real(path, dark_img):
    with rawpy.imread(path) as raw:
        img = raw.raw_image_visible.astype(np.float32)
        red = img[1::2, 0::2]
        dark_red = dark_img[1::2, 0::2]
        red = red - dark_red
        red = np.maximum(red, 0)
        return red


def leer_dark_folder(folder_path):
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".dng")])
    dark_stack = []

    for f in files:
        path = os.path.join(folder_path, f)
        with rawpy.imread(path) as raw:
            img = raw.raw_image_visible.astype(np.float32)
            dark_stack.append(img)

    dark_stack = np.array(dark_stack)
    dark_mean = np.mean(dark_stack, axis=0).astype(np.float32)
    return dark_mean


def calcular_matriz_intensidades(imagenes, y1, y2, x1, x2):
    I_matrix = np.zeros((6, 6), dtype=np.float32)
    for j in range(6):          # PSG
        for i in range(6):      # PSA
            roi = imagenes[j][i][y1:y2, x1:x2]
            I_matrix[j, i] = np.mean(roi)
    return I_matrix


def calibrar_A(I_matrix, W_norm):
    P_list = []
    for i in range(6):  # columna = PSA fijo
        I_i = I_matrix[:, i]
        a_i = np.linalg.pinv(W_norm) @ I_i
        P_list.append(a_i)
    A = np.vstack(P_list)   # 6 x 4
    return A


def validar_stokes(I_matrix, W_norm, A, lam=1e-3):
    col_norms = np.linalg.norm(A, axis=0)
    A_scaled = A / col_norms

    errores = []
    dop_list = []
    S_rec_list = []

    # ===== métricas globales =====
    error_global_num = 0.0
    error_global_den = 0.0

    for idx in range(6):
        S_theory = W_norm[idx].copy()
        I_meas = I_matrix[idx, :]

        S_scaled = np.linalg.solve(
            A_scaled.T @ A_scaled + lam * np.eye(4),
            A_scaled.T @ I_meas
        )

        S_rec = S_scaled / col_norms

        # normalización
        S_rec = S_rec / S_rec[0]
        S_theory = S_theory / S_theory[0]

        error_rel = np.linalg.norm(S_rec - S_theory) / np.linalg.norm(S_theory)
        dop = np.sqrt(S_rec[1]**2 + S_rec[2]**2 + S_rec[3]**2)

        errores.append(error_rel)
        dop_list.append(dop)
        S_rec_list.append(S_rec)

        # ===== NUEVO: acumulado global =====
        error_global_num += np.linalg.norm(S_rec - S_theory)
        error_global_den += np.linalg.norm(S_theory)

    error_medio = float(np.mean(errores))
    error_global = error_global_num / error_global_den

    return error_medio, error_global, errores, dop_list, S_rec_list, A, col_norms

# =========================================================
# DATOS
# =========================================================

dark_folder = ""
dark_img = leer_dark_folder(dark_folder)

W = np.array([
    [1, 1, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 0, -1],
    [1, 0, -1, 0],
    [1, 0, 1, 0],
    [1, -1, 0, 0]
], dtype=np.float32)

W_norm = W / W[:, 0][:, None]

base_folder = ""

order = [
    "Horizontal",
    "PCD",
    "PCI",
    "menos45",
    "mas45",
    "Vertical"
]

carpetas = [os.path.join(base_folder, name) for name in order]

# =========================================================
# CARGA DE IMÁGENES UNA SOLA VEZ
# =========================================================

imagenes = []
for folder in carpetas:
    files = sorted([f for f in os.listdir(folder) if f.endswith(".dng")])
    paths = [os.path.join(folder, f) for f in files]
    imgs_folder = [leer_rojo_real(p, dark_img) for p in paths]
    imagenes.append(imgs_folder)

# =========================================================
# ROI INICIAL
# =========================================================

ref_img = imagenes[0][0]
print("Selecciona UNA ROI para todo el experimento")

mask = ref_img > 0.6 * np.max(ref_img)
ys, xs = np.where(mask)

pad = 20
y1 = max(0, ys.min() - pad)
y2 = min(ref_img.shape[0], ys.max() + pad)
x1 = max(0, xs.min() - pad)
x2 = min(ref_img.shape[1], xs.max() + pad)

output_roi_folder = "ROI_debug"
os.makedirs(output_roi_folder, exist_ok=True)

# guardar ROI inicial
guardar_roi_debug(ref_img, y1, y2, x1, x2, os.path.join(output_roi_folder, "ROI_inicial.png"), title="ROI inicial")

# =========================================================
# BUCLE DE BÚSQUEDA DE LA MEJOR ROI
# =========================================================

max_iter = 1000
lam = 1E-3

best_error = np.inf
best_roi = (y1, y2, x1, x2)
best_results = None

# guardamos también el mejor error anterior para ver evolución
error_anterior = np.inf

alto, ancho = ref_img.shape
log_file = open("log_iteraciones.txt", "w")
log_file.write("Iteración | ROI | error | best_error\n")
for it in range(max_iter):
    # matriz de intensidades para la ROI actual
    I_matrix = calcular_matriz_intensidades(imagenes, y1, y2, x1, x2)

    # calibración
    A = calibrar_A(I_matrix, W_norm)


    I_pred = A @ W_norm.T
    error_ajuste_A = np.linalg.norm(I_matrix - I_pred) / np.linalg.norm(I_matrix)

    # validación
    error_medio, error_global, errores, dop_list, S_rec_list, A, col_norms = validar_stokes(
        I_matrix, W_norm, A, lam=lam
    )

    print(f"Iter {it:4d} | ROI=({y1}:{y2},{x1}:{x2}) | "
        f"error_medio={error_medio:.6f} | error_global={error_global:.6f} | best={best_error:.6f}")
    log_file.write(
    f"Iter {it:4d} | ROI=({y1}:{y2},{x1}:{x2}) | "
    f"error={error_medio:.6f} | best={best_error:.6f}\n"
    )
    log_file.flush()
    # guardar mejor ROI
    if error_medio < best_error:
        best_error = error_medio
        best_roi = (y1, y2, x1, x2)
        best_results = {
            "I_matrix": I_matrix.copy(),
            "A": A.copy(),
            "errores": errores.copy(),
            "dop_list": dop_list.copy(),
            "S_rec_list": S_rec_list.copy(),
            "col_norms": col_norms.copy()
        }
        guardar_roi_debug(
            ref_img, y1, y2, x1, x2,
            os.path.join(output_roi_folder, "ROI_mejor_actual.png"),
            title=f"Mejor ROI (it={it})"
        )


    error_anterior = error_medio

    # expandir ROI 1 píxel por lado
    nueva_y1 = max(0, y1 - 1)
    nueva_y2 = min(alto, y2 + 1)
    nueva_x1 = max(0, x1 - 1)
    nueva_x2 = min(ancho, x2 + 1)

    # si ya no puede crecer más, parar
    if (nueva_y1, nueva_y2, nueva_x1, nueva_x2) == (y1, y2, x1, x2):
        print("La ROI ya no puede seguir creciendo.")
        break

    y1, y2, x1, x2 = nueva_y1, nueva_y2, nueva_x1, nueva_x2

print("\n================ RESULTADO FINAL ================")
print("Mejor ROI:", best_roi)
print("Mejor error medio:", best_error)

# =========================================================
# VALIDACIÓN FINAL
# =========================================================

y1, y2, x1, x2 = best_roi
I_matrix = best_results["I_matrix"]
A = best_results["A"]
np.set_printoptions(precision=4, suppress=True)

print("\n=== MEJOR MATRIZ A (calibración final) ===")
print("Filas = PSA | Columnas = [S0, S1, S2, S3]\n")

for i, fila in enumerate(A):
    print(f"PSA {i}: ", " ".join(f"{val:8.4f}" for val in fila))

    A_scaled = A / col_norms

print("\n=== MATRIZ A ESCALADA ===")

for i, fila in enumerate(A_scaled):
    print(f"PSA {i}: ", " ".join(f"{val:8.4f}" for val in fila))
col_norms = best_results["col_norms"]
errores = best_results["errores"]
dop_list = best_results["dop_list"]
S_rec_list = best_results["S_rec_list"]

log_file.close()

print("\n=== MATRIZ DE INTENSIDADES FINAL ===")
print(I_matrix)

print("\n=== RESULTADOS DE LA MEJOR ROI ===")
print(f"Mejor ROI encontrada: {best_roi}")
print(f"Error de ajuste de la matriz A: {error_ajuste_A * 100:.2f}%")
print(f"Error medio de reconstrucción Stokes: {best_error:.6f}")

print("\n================ VALIDACIÓN GLOBAL FINAL =================")
for idx in range(6):
    print(f"\n--- PSG {idx} ---")
    print("S teórico:     ", W_norm[idx])
    print("S reconstruido:", S_rec_list[idx])
    print("Error relativo:", errores[idx])
    print("DoP:", dop_list[idx])

print("\nCond(A):", np.linalg.cond(A))
print("Rank(A):", np.linalg.matrix_rank(A))

print(np.mean(np.abs(A[:, 0])))
print(np.mean(np.abs(A[:, 1])))
print(np.mean(np.abs(A[:, 2])))
print(np.mean(np.abs(A[:, 3])))

print(np.linalg.norm(A[:, 0]))
print(np.linalg.norm(A[:, 1]))
print(np.linalg.norm(A[:, 2]))
print(np.linalg.norm(A[:, 3]))