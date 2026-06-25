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

# =========================================================
# 1. Funciones
# =========================================================


dark_folder = ""
dark_img = leer_dark_folder(dark_folder)

# =========================================================
# 1. DATOS DE ENTRADA
# =========================================================

#W de nueva calibración
W = np.array([
    [1, 1, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 0, -1],
    [1, 0, -1, 0],
    [1, 0, 1, 0],
    [1, -1, 0, 0]
])



W_norm = W / W[:,0][:, None]

# =========================================================
# Extraer I
# =========================================================

base_folder = ""

order = [
    "Horizontal",
    "PCD",
    "PCI",
    "menos45",
    "mas45",
    "Vertical"
]

carpetas = [
    os.path.join(base_folder, name)
    for name in order
]

I_matrix = np.zeros((6,6))

# =========================================================
# ROI GLOBAL
# =========================================================

ref_folder = carpetas[0]
files_ref = sorted([f for f in os.listdir(ref_folder) if f.endswith(".dng")])
path_ref = os.path.join(ref_folder, files_ref[0])

img_ref = leer_rojo_real(path_ref, dark_img)

print("Selecciona UNA ROI para todo el experimento")

#y1, y2, x1, x2 = seleccionar_roi(img_ref)

#zona iluminada
mask = img_ref > 0.6 * np.max(img_ref)
ys, xs = np.where(mask)

pad = 20
y1 = max(0, ys.min() - pad)
y2 = min(img_ref.shape[0], ys.max() + pad)
x1 = max(0, xs.min() - pad)
x2 = min(img_ref.shape[1], xs.max() + pad)

output_roi_folder = "ROI_debug"
os.makedirs(output_roi_folder, exist_ok=True)

for j, folder in enumerate(carpetas):

    # PSA
    files = sorted([f for f in os.listdir(folder) if f.endswith(".dng")])
    paths = [os.path.join(folder, f) for f in files]

    """
    img_ref = leer_raw_corrected_rojo(paths[0], dark_img)
    print(f"ROI en {os.path.basename(folder)}")
    y1, y2, x1, x2 = seleccionar_roi(img_ref)
    """

    for i, p in enumerate(paths):

        # PSG
        img = leer_rojo_real(p, dark_img)
        roi = img[y1:y2, x1:x2]  # ← usa ROI GLOBAL

        if i == 0:  # 1 imagen por configuración PSA
            out_path = os.path.join(
                output_roi_folder,
                f"{order[j]}_ROI.png"
            )
            guardar_roi_debug(
                img,
                y1,
                y2,
                x1,
                x2,
                out_path,
                title=order[j]
            )

        I_matrix[j, i] = np.mean(roi)

        #I_matrix[j, :] = I_matrix[j, :] / np.max(I_matrix[j, :])

print("\n=== MATRIZ DE INTENSIDADES I_matrix ===")
print(I_matrix)
print([os.path.basename(c) for c in carpetas])
print([f for f in files])

# =========================================================
# 3. CÁLCULO A
# =========================================================

P_list = []

for i in range(6):
    I_i = I_matrix[:, i]      # columna: todos los PSG para un PSA fijo
    a_i = np.linalg.pinv(W_norm) @ I_i
    P_list.append(a_i)

A = np.vstack(P_list)         # 6 x 4# (6 x 4)

np.set_printoptions(precision=4, suppress=True)

print("\n=== MATRIZ A (calibración) ===")
print(A)

print("A shape:", A.shape)
print("A shape:", A.shape)

idx = 0  # PSG Horizontal
S_test = W_norm[idx]  # Stokes del PSG
I_pred = A @ S_test  # lo que A predice
I_real = I_matrix[:, idx]  # lo que mido

error = np.linalg.norm(I_pred - I_real) / np.linalg.norm(I_real)

print("Error relativo A vs medición:", error)
print("\nCond(A):", np.linalg.cond(A))
print("Rank(A):", np.linalg.matrix_rank(A))

# =========================================
# ESCALADO DE COLUMNAS DE A
# =========================================

col_norms = np.linalg.norm(A, axis=0)
A_scaled = A / col_norms

lam = 1e-3

# =========================================================
# VALIDACIÓN DE STOKES
# =========================================================

print("\n================ VALIDACIÓN GLOBAL =================")
errors_num = 0.0
errors_den = 0.0
for idx in range(6):

    # recorrer todos los PSG
    S_theory = W_norm[idx]
    I_meas = I_matrix[idx, :]

    # reconstrucción
    S_scaled = np.linalg.solve(
        A_scaled.T @ A_scaled + lam * np.eye(4),
        A_scaled.T @ I_meas
    )

    S_rec = S_scaled / col_norms

    # normalización
    S_rec = S_rec / S_rec[0]
    S_theory = S_theory / S_theory[0]
    errors_num += np.linalg.norm(S_rec - S_theory)
    errors_den += np.linalg.norm(S_theory)
    error = S_rec - S_theory
    error_rel = np.linalg.norm(error) / np.linalg.norm(S_theory)

    dop = np.sqrt(S_rec[1]**2 + S_rec[2]**2 + S_rec[3]**2)

    print(f"\n--- PSG {idx} ---")
    print("S teórico: ", S_theory)
    print("S reconstruido:", S_rec)
    print("Error relativo:", error_rel)
    print("DoP:", dop)
    E_global = errors_num / errors_den

    print("\n================ ERROR GLOBAL =================")
    print("Error relativo global:", E_global)

