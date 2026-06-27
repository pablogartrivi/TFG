import numpy as np
import rawpy
from pathlib import Path
import cv2


def generar_rois(H, W, step=20, min_size=30, max_size=None):
    cx, cy = W // 2, H // 2

    if max_size is None:
        max_size = min(H, W) // 2

    rois = []

    for size in range(min_size, max_size, step):
        x0 = max(0, cx - size)
        x1 = min(W, cx + size)
        y0 = max(0, cy - size)
        y1 = min(H, cy + size)

        rois.append((x0, y0, x1, y1))

    return rois


def evaluar_roi(images, roi, A, S_theo, dark_img):
    I_vec = construir_I(images, roi)

    I_vec = I_vec / np.mean(I_vec)
    S = estimar_stokes(A, I_vec)

    S_err = np.linalg.norm(S - S_theo) / np.linalg.norm(S_theo)

    return S_err, S


def buscar_mejor_roi(images, A, S_theo, dark_img, max_iter=50, step=10):
    H, W = images[0].shape

    cx, cy = W // 2, H // 2
    size = min(H, W) // 6

    best_roi = (cx - size, cy - size, cx + size, cy + size)

    def score(roi):
        I_vec = construir_I(images, roi)
        S = estimar_stokes(A, I_vec)

        err = np.linalg.norm(S - S_theo) / np.linalg.norm(S_theo)

        x0, y0, x1, y1 = roi
        area = (x1 - x0) * (y1 - y0)

        penalty = 1e-6 / area

        return err + penalty, S

    best_error, best_S = score(best_roi)
    print("\n=== STOKES INICIAL (ANTES DE MINIMIZACIÓN) ===")
    print(best_S)

    for _ in range(max_iter):

        candidates = []

        x0, y0, x1, y1 = best_roi
        shifts = [-step, 0, step]

        for dx in shifts:
            for dy in shifts:
                for ds in [-step, step]:

                    nx0 = max(0, x0 - ds + dx)
                    ny0 = max(0, y0 - ds + dy)
                    nx1 = min(W, x1 + ds + dx)
                    ny1 = min(H, y1 + ds + dy)

                    if nx1 <= nx0 or ny1 <= ny0:
                        continue

                    roi = (nx0, ny0, nx1, ny1)
                    err, S = score(roi)

                    candidates.append((err, S, roi))

        if not candidates:
            break

        err, S, roi = min(candidates, key=lambda x: x[0])

        if err < best_error:
            best_error = err
            best_S = S
            best_roi = roi
        else:
            break

    return best_roi, best_S, best_error

# =========================================================
# LECTURA CANAL ROJO
# =========================================================
def leer_rojo_real(path, dark_img):
    with rawpy.imread(str(path)) as raw:
        img = raw.raw_image_visible.astype(np.float32)
        red = img[1::2, 0::2]
        dark_red = dark_img[1::2, 0::2]
        red = red - dark_red
        red = np.maximum(red, 0)
        return red



def crear_master_dark(folder):
    folder = Path(folder)
    files = sorted(folder.glob("*.dng"))

    stack = []
    for f in files:
        with rawpy.imread(str(f)) as raw:
            img = raw.raw_image_visible.astype(np.float32)
            stack.append(img)

    return np.mean(stack, axis=0)


# =========================================================
# ROI SIMPLE 
# =========================================================
def seleccionar_roi(image):
    roi = cv2.selectROI("Selecciona ROI", image, False, False)
    cv2.destroyAllWindows()

    x, y, w, h = roi
    return (x, y, x + w, y + h)


def intensidad_media(img, roi):
    x0, y0, x1, y1 = roi
    patch = img[y0:y1, x0:x1]

    p5, p95 = np.percentile(patch, [5, 95])
    patch = patch[(patch >= p5) & (patch <= p95)]

    return np.mean(patch)


def construir_I(images, roi):
    I = [intensidad_media(img, roi) for img in images]
    return np.array(I, dtype=np.float64).reshape(-1, 1)


# =========================================================
# STOKES
# =========================================================

def estimar_stokes(A, I_vec, lam=1e-3):

    A = np.asarray(A, dtype=np.float64)

    col_norms = np.linalg.norm(A, axis=0)
    A_scaled = A / col_norms

    S_scaled = np.linalg.solve(
        A_scaled.T @ A_scaled + lam * np.eye(4),
        A_scaled.T @ I_vec
    )

    S_hat = S_scaled / col_norms.reshape(-1, 1)

    # normalización
    S0 = S_hat[0, 0]
    S_hat = S_hat / S0
    S_hat[0, 0] = 1.0

    return S_hat


# =========================================================
# 5) MAIN
# =========================================================
if __name__ == "__main__":

    # -------------------------
    # MATRIZ A
    # -------------------------
    A = np.array([

    ], dtype=np.float64)

    A = A / np.mean(A[:, 0])

    S_theo = np.array( dtype=np.float64)

    folder = Path("")
    dark_folder = Path("")

    dark_img = crear_master_dark(dark_folder)

    image_paths = sorted(folder.glob("*.dng"))

    if len(image_paths) != 6:
        raise ValueError("La carpeta debe tener exactamente 6 imágenes")

    # -------------------------
    # CARGA IMÁGENES
    # -------------------------
    images = [leer_rojo_real(p, dark_img) for p in image_paths]


    best_roi, S, S_error_rel = buscar_mejor_roi(images, A, S_theo, dark_img)

    S_error = np.linalg.norm(S - S_theo)


print("\n=== STOKES TEÓRICO ===")
print(S_theo)

print("\n=== MEJOR ROI ===")
print(best_roi)

print("\n=== STOKES RESULTANTE ===")
print(S)

print("\n=== ERROR ===")
print("Error absoluto:", S_error)
print("Error relativo:", S_error_rel)
