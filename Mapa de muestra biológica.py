import os
import rawpy
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import shift as ndi_shift, gaussian_filter, convolve, zoom

from skimage.registration import phase_cross_correlation
from skimage import exposure
from skimage.restoration import denoise_bilateral

import matplotlib.colors as mcolors

# =========================================================
# FUNCIONES
# =========================================================


def leer_rgb(path):
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            output_bps=16
        )
    return rgb

def auto_limits(img, pmin=2, pmax=98):
    """
    Límites robustos para imágenes/mapas.
    Ignora NaN e Inf y usa percentiles.
    """
    data = np.asarray(img)
    data = data[np.isfinite(data)]

    if data.size == 0:
        return 0.0, 1.0

    vmin, vmax = np.percentile(data, [pmin, pmax])

    # Evita casos degenerados
    if np.isclose(vmin, vmax):
        eps = 1e-6 if vmin == 0 else 0.01 * abs(vmin)
        vmin -= eps
        vmax += eps

    return float(vmin), float(vmax)


def auto_symmetric_limits(img, pmax=98):
    """
    Límites simétricos alrededor de 0 para mapas con signo.
    """
    data = np.asarray(img)
    data = data[np.isfinite(data)]

    if data.size == 0:
        return -1.0, 1.0

    lim = np.percentile(np.abs(data), pmax)

    if np.isclose(lim, 0):
        lim = 1.0

    return float(-lim), float(lim)


def leer_rojo_real(path, dark_img):
    with rawpy.imread(path) as raw:
        img = raw.raw_image_visible.astype(np.float32)

        # 1. Restamos el Dark Frame a resolución completa
        clean_raw = img - dark_img
        clean_raw = np.maximum(clean_raw, 0)

        # 2. Aislamos los píxeles rojos en su posición original
        full_red = np.zeros_like(clean_raw)
        full_red[1::2, 0::2] = clean_raw[1::2, 0::2]

        # 3. Kernel de interpolación bilineal para patrón Bayer
        kernel = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype=np.float32) / 4.0

        # 4. Convolución para rellenar huecos
        red_demosaiced = convolve(full_red, kernel, mode='reflect')

        return red_demosaiced


def leer_dark_folder(folder_path):
    files = sorted([
        f for f in os.listdir(folder_path)
        if f.endswith(".dng")
    ])

    dark_stack = []

    for f in files:
        path = os.path.join(folder_path, f)
        with rawpy.imread(path) as raw:
            img = raw.raw_image_visible.astype(np.float32)
            dark_stack.append(img)

    dark_stack = np.array(dark_stack)
    dark_mean = np.mean(dark_stack, axis=0).astype(np.float32)

    return dark_mean


def alinear_stack(imagenes, upsample_factor=20):
    ref = imagenes[0]
    shifts = [(0.0, 0.0)]

    # 1. Desplazamientos sub-píxel en la resolución base
    ref_norm = ref / (np.max(ref) + 1e-12)

    for img in imagenes[1:]:
        img_norm = img / (np.max(img) + 1e-12)

        shift, error, diffphase = phase_cross_correlation(
            ref_norm,
            img_norm,
            upsample_factor=upsample_factor
        )
        shifts.append(shift)

    # 2. Reconstrucción a súper-resolución (factor x2)
    imgs_alineadas_sr = []
    factor_sr = 2

    for img, shift in zip(imagenes, shifts):
        img_scaled = zoom(img, zoom=factor_sr, order=3)
        shift_sr = [s * factor_sr for s in shift]

        img_aligned = ndi_shift(
            img_scaled,
            shift=shift_sr,
            order=3,
            mode='constant',
            cval=0.0,
            prefilter=True
        )
        imgs_alineadas_sr.append(img_aligned)

    return np.array(imgs_alineadas_sr), shifts


def preparar_fondo_biologico(img):
    img = np.nan_to_num(img)

    # normalización robusta
    p1, p99 = np.percentile(img, (1, 99))
    img = np.clip((img - p1) / (p99 - p1 + 1e-12), 0, 1)

    # reducción de ruido preservando bordes
    img = denoise_bilateral(
        img,
        sigma_color=0.05,
        sigma_spatial=3
    )

    # CLAHE
    img = exposure.equalize_adapthist(
        img,
        clip_limit=0.03
    )

    # gamma
    img = exposure.adjust_gamma(img, gamma=0.8)

    return img


# =========================================================
# 1. DARK FRAMES
# =========================================================

dark_folder = "C:/Users/Usuario/Desktop/TFG/Estudio RAW/Estudio RAW/Dark_frames_1,200"
dark_img = leer_dark_folder(dark_folder)


# =========================================================
# 2. MATRIZ A CALIBRADA
# =========================================================

A = np.array([
    [283.3268, -45.3888,  101.5007, -264.2766],
    [283.1742, -14.2220,  -78.8413, -279.0558],
    [281.7852, -240.0363, -100.9310, -110.2365],
    [281.8015, -229.9635,   95.2770,  110.0863],
    [283.7793,  -8.7202,    47.3450,  281.6459],
    [284.8248, -59.3928,  -124.6307,  244.0174]
], dtype=np.float32)

A_pinv = np.linalg.pinv(A)


# =========================================================
# 3. CARPETA DE IMÁGENES
# =========================================================

folder_LHP = "C:/Users/Usuario/Desktop/analisis_melanoma/prueba54"

files = sorted([
    f for f in os.listdir(folder_LHP)
    if f.endswith(".dng")
])

if len(files) != 6:
    raise ValueError("Debe haber EXACTAMENTE 6 imágenes")

paths = [os.path.join(folder_LHP, f) for f in files]

rgb_img = leer_rgb(paths[0])
# =========================================================
# 4. CARGAR IMÁGENES
# =========================================================

print("\nCargando imágenes...")
imgs = []
for p in paths:
    img = leer_rojo_real(p, dark_img)
    imgs.append(img)

imgs = np.array(imgs)

imgs = gaussian_filter(imgs, sigma=(0, 0.8, 0.8))
# =========================================================
# 4.5 SELECCIÓN MANUAL DE ROI
# =========================================================

print("\nPor favor, selecciona la ROI en la ventana emergente...")
preview = exposure.equalize_adapthist(
    imgs[0] / (np.max(imgs[0]) + 1e-12),
    clip_limit=0.05
)

fig_roi, ax_roi = plt.subplots(figsize=(10, 8))
ax_roi.imshow(preview, cmap='gray')
ax_roi.set_title("Haz clic en 2 esquinas opuestas para seleccionar la ROI")
plt.show(block=False)

pts = plt.ginput(2, timeout=0)
plt.close(fig_roi)

if len(pts) == 2:
    (x1, y1), (x2, y2) = pts
    x1, x2 = int(min(x1, x2)), int(max(x1, x2))
    y1, y2 = int(min(y1, y2)), int(max(y1, y2))

    imgs = imgs[:, y1:y2, x1:x2]
    rgb_img = rgb_img[y1:y2, x1:x2]
    print(f"-> ROI seleccionada y recortada: Y[{y1}:{y2}], X[{x1}:{x2}]")
else:
    print("-> No se seleccionaron 2 puntos. Se procesará la imagen completa.")


# =========================================================
# 5. ALINEACIÓN
# =========================================================

print("\nAlineando stack (esto será mucho más rápido ahora)...")
imgs, shifts = alinear_stack(imgs, upsample_factor=20)
rgb_img = zoom(rgb_img, (2, 2, 1), order=3)

print("\nDesplazamientos aplicados:")
for k, s in enumerate(shifts):
    print(f"Imagen {k}: shift = {s}")

H, W_img = imgs.shape[1], imgs.shape[2]


# =========================================================
# 6. RECONSTRUCCIÓN STOKES
# =========================================================

imgs_reshaped = imgs.reshape(6, -1)
lambda_reg = 1

A_reg = A.T @ np.linalg.inv(A @ A.T + lambda_reg * np.eye(6))
S_all = A_reg @ imgs_reshaped

S0_raw = S_all[0, :]
S1_raw = S_all[1, :]
S2_raw = S_all[2, :]
S3_raw = S_all[3, :]


# =========================================================
# 7. FILTRADO POLARIMÉTRICO
# =========================================================

sigma_pol = 1.2

S0_s = gaussian_filter(S0_raw.reshape(H, W_img), sigma_pol)
S1_s = gaussian_filter(S1_raw.reshape(H, W_img), sigma_pol)
S2_s = gaussian_filter(S2_raw.reshape(H, W_img), sigma_pol)
S3_s = gaussian_filter(S3_raw.reshape(H, W_img), sigma_pol)

S0_raw = S0_s.ravel()
S1_raw = S1_s.ravel()
S2_raw = S2_s.ravel()
S3_raw = S3_s.ravel()


# =========================================================
# 8. NORMALIZACIÓN
# =========================================================

sigma_noise = np.std(S0_raw)
threshold = 3 * sigma_noise
valid = S0_raw > threshold

eps = 1e-12
snr = S0_raw / (np.std(S0_raw) + eps)

valid = snr > 3

S1_norm = np.full_like(S1_raw, np.nan)
S2_norm = np.full_like(S2_raw, np.nan)
S3_norm = np.full_like(S3_raw, np.nan)

eps = 1e-6 * np.max(S0_raw)

S0_safe = np.where(S0_raw > eps, S0_raw, np.nan)

S1_norm = S1_raw / S0_safe
S2_norm = S2_raw / S0_safe
S3_norm = S3_raw / S0_safe


# =========================================================
# 9. MAPAS 2D
# =========================================================

I_map = S0_raw.reshape(H, W_img)
S1_map = S1_norm.reshape(H, W_img)
S2_map = S2_norm.reshape(H, W_img)
S3_map = S3_norm.reshape(H, W_img)


# =========================================================
# 10. AoLP Y ELIPTICIDAD
# =========================================================

azimuth_map = 0.5 * np.arctan2(S2_map, S1_map)

ellipticity_map = 0.5 * np.arcsin(
    np.clip(S3_map, -1, 1)
)

azimuth_deg = np.degrees(azimuth_map)
ellipticity_deg = np.degrees(ellipticity_map)


# =========================================================
# 11. DoLP Y DoP
# =========================================================

DoLP_map = np.sqrt(S1_map**2 + S2_map**2)
DoLP_map = np.clip(DoLP_map, 0, 1)

DoP_map = np.sqrt(
    S1_map**2 +
    S2_map**2 +
    S3_map**2
)
DoP_map = np.clip(DoP_map, 0, 1)


# =========================================================
# 12. LIMPIEZA
# =========================================================

I_map = np.nan_to_num(I_map)
DoLP_map = np.nan_to_num(DoLP_map)
DoP_map = np.nan_to_num(DoP_map)
azimuth_deg = np.nan_to_num(azimuth_deg)
ellipticity_deg = np.nan_to_num(ellipticity_deg)


# =========================================================
# 13. MÁSCARA POLARIMÉTRICA
# =========================================================

mask_pol = (
    (DoLP_map > 0.0) &
    (I_map > np.percentile(I_map, 0))
)

"""
alpha_pol = np.clip(
    (DoLP_map - 0.25) / 0.75,
    0,
    0.85
)
"""
alpha_pol = np.clip((DoLP_map - 0.1) / 0.8, 0, 1) 
alpha_pol = np.where(mask_pol, alpha_pol, 0)


# =========================================================
# 14. AZIMUT CON CONTRASTE VISUAL FUERTE
# =========================================================

az_show = azimuth_deg.copy()
az_show = np.nan_to_num(az_show)
az_show = gaussian_filter(az_show, sigma=0.8)
az_show = np.where(mask_pol, az_show, np.nan)


# =========================================================
# 15. ELIPTICIDAD MOSTRABLE
# =========================================================

el_show = np.where(mask_pol, ellipticity_deg, np.nan)


# =========================================================
# 16. FONDO BIOLÓGICO Y LÍMITES AUTOMÁTICOS
# =========================================================

I_disp = preparar_fondo_biologico(I_map)

I_vmin, I_vmax = auto_limits(I_disp, pmin=2, pmax=98)

DoLP_vmin, DoLP_vmax = auto_limits(DoLP_map, pmin=1, pmax=99)
DoLP_vmin = max(0, DoLP_vmin)
DoLP_vmax = min(1, DoLP_vmax)

AoLP_vmin, AoLP_vmax = -90, 90

el_vmin, el_vmax = auto_symmetric_limits(el_show, pmax=98)


# =========================================================
# 17. FIGURAS PRINCIPALES
# =========================================================

# RGB
plt.figure()
plt.imshow(rgb_img / np.max(rgb_img))
plt.title("RGB")
plt.axis("off")
plt.show()

# Intensidad
plt.figure()
plt.imshow(I_disp, cmap='gray', vmin=I_vmin, vmax=I_vmax)
plt.title("Intensidad")
plt.axis("off")
plt.show()

# DoLP
plt.figure()
plt.imshow(I_disp, cmap='gray')
plt.imshow(DoLP_map, cmap='nipy_spectral', vmin=0.2, vmax=1, alpha=alpha_pol)
plt.title("DoP")
plt.axis("off")
plt.colorbar(label="DoP")
plt.show()

# AoLP
plt.figure()
plt.imshow(I_disp, cmap='gray')
plt.imshow(az_show, cmap='hsv', vmin=-90, vmax=-70, alpha=alpha_pol)
plt.title("AoP")
plt.axis("off")
plt.colorbar(label="Ángulo (°)")
plt.show()

# Elipticidad
plt.figure()
plt.imshow(I_disp, cmap='gray')
plt.imshow(el_show, cmap='seismic', vmin=30, vmax=45, alpha=alpha_pol)
plt.title("Elipticidad")
plt.axis("off")
plt.colorbar(label="Elipticidad (°)")
plt.show()

print("S0 min =", np.min(S0_raw))
print("S0 max =", np.max(S0_raw))

print("Pixels válidos =", np.sum(valid))
print("Pixels totales =", len(valid))

print("Porcentaje válido =", 100*np.mean(valid))

# =========================================================
# FIN
# =========================================================

print("\nReconstrucción polarimétrica biomédica completada.")
