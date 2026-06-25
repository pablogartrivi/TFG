import os
import rawpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift as ndi_shift
from skimage.registration import phase_cross_correlation
from matplotlib import colors
from sklearn.cluster import KMeans
from skimage import exposure
# =========================================================
# FUNCIONES
# =========================================================

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

def alinear_stack(imagenes, upsample_factor=20):
    """
    Alinea todas las imágenes al primer frame usando correlación de fase.
    Devuelve:
        imgs_alineadas: np.array con shape (N, H, W)
        shifts: lista de desplazamientos (dy, dx)
    """
    ref = imagenes[0]
    imgs_alineadas = [ref]
    shifts = [(0.0, 0.0)]

    # opcional: usar una versión normalizada para estimar mejor el shift
    ref_norm = ref / (np.max(ref) + 1e-12)

    for img in imagenes[1:]:
        img_norm = img / (np.max(img) + 1e-12)

        shift, error, diffphase = phase_cross_correlation(
            ref_norm, img_norm, upsample_factor=upsample_factor
        )

        img_aligned = ndi_shift(
            img,
            shift=shift,
            order=1,
            mode='constant',
            cval=0.0,
            prefilter=False
        )

        imgs_alineadas.append(img_aligned)
        shifts.append(shift)

    return np.array(imgs_alineadas), shifts
# =========================================================
# 1. DARK
# =========================================================

dark_folder = "C:/Users/Usuario/Desktop/TFG/Estudio RAW/Estudio RAW/Dark_frames_1,200"
dark_img = leer_dark_folder(dark_folder)

# =========================================================
# 2. MATRIZ A (YA CALIBRADA)
# 👉 PEGA AQUÍ TU MATRIZ A (6x4)
# =========================================================

A = np.array([
    [283.3086,  -45.3328,  101.5287, -263.3156],
    [283.3528,  -14.1695,  -78.8150, -279.3082],
    [282.7534, -239.9890, -100.9073, -108.3411],
    [282.4090, -229.9181,   95.2997,  112.8620],
    [283.7293,   -8.6898,   47.3602,  281.9696],
    [285.2304,  -59.3687, -124.6186,  243.9880]
], dtype=np.float32)

# Pseudoinversa
A_pinv = np.linalg.pinv(A)

# =========================================================
# 3. CARPETA LHP (SOLO 6 IMÁGENES)
# =========================================================

folder_LHP = "C:/Users/Usuario/Desktop/TFG/Estudio RAW/Estudio RAW/Estudioestructuradiedrica" \
#+retardador
#+retardador"  # ← AJUSTA                  12
#folder_LHP = "C:/Users/Usuario/Desktop/analisis_melanoma/prueba44" # ← AJUSTA

files = sorted([f for f in os.listdir(folder_LHP) if f.endswith(".dng")])

if len(files) != 6:
    raise ValueError("Debe haber EXACTAMENTE 6 imágenes (una por estado del PSA)")

paths = [os.path.join(folder_LHP, f) for f in files]

# =========================================================
# 4. CARGAR IMÁGENES
# =========================================================

imgs = []
for p in paths:
    img = leer_rojo_real(p, dark_img)
    imgs.append(img)

imgs = np.array(imgs)  # (6, H, W)

# =========================================================
# ALINEACIÓN DE LAS 6 IMÁGENES
# =========================================================
imgs, shifts = alinear_stack(imgs, upsample_factor=20)

print("Desplazamientos aplicados (dy, dx) respecto a la primera imagen:")
for k, s in enumerate(shifts):
    print(f"Imagen {k}: shift = {s}")

H, W_img = imgs.shape[1], imgs.shape[2]


# =========================================================
# USAR TODA LA IMAGEN (SIN ROI)
# =========================================================

H, W_img = imgs.shape[1], imgs.shape[2]

print("Usando TODA la imagen para reconstrucción")
print(f"Dimensiones: {H} x {W_img}")
from matplotlib.colors import TwoSlopeNorm

# =========================================================
# 5. RECONSTRUCCIÓN PIXEL A PIXEL
# =========================================================

imgs_reshaped = imgs.reshape(6, -1)      # (6, Npix)
S_all = A_pinv @ imgs_reshaped           # (4, Npix)

# Guardar intensidad original antes de normalizar
S0_raw = S_all[0, :].copy()
S1_raw = S_all[1, :].copy()
S2_raw = S_all[2, :].copy()
S3_raw = S_all[3, :].copy()

threshold = 0.05 * np.max(S0_raw)

valid = S0_raw > threshold

# Normalizar parámetros de Stokes
S1_norm = np.full_like(S1_raw, np.nan)
S2_norm = np.full_like(S2_raw, np.nan)
S3_norm = np.full_like(S3_raw, np.nan)

S1_norm[valid] = S1_raw[valid] / S0_raw[valid]
S2_norm[valid] = S2_raw[valid] / S0_raw[valid]
S3_norm[valid] = S3_raw[valid] / S0_raw[valid]

# Mapas 2D
I_map = S0_raw.reshape(H, W_img)
S1_map = S1_norm.reshape(H, W_img)
S2_map = S2_norm.reshape(H, W_img)
S3_map = S3_norm.reshape(H, W_img)



# Azimut y elipticidad
azimuth_map = 0.5 * np.arctan2(S2_map, S1_map)         # rad
ellipticity_map = 0.5 * np.arcsin(np.clip(S3_map, -1, 1))  # rad

# Pasar a grados para que se entienda mejor
azimuth_deg = np.degrees(azimuth_map)
ellipticity_deg = np.degrees(ellipticity_map)



# =========================================================
# FUNCIONES DE VISUALIZACIÓN
# =========================================================

def preparar_fondo(img, pmin=1, pmax=99, gamma=0.7, clahe=True):
    """
    Mejora el contraste del fondo para que la imagen se vea más clara.
    - recorte por percentiles
    - corrección gamma
    - equalización adaptativa opcional
    """
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    valid = img > 0
    if np.any(valid):
        lo, hi = np.percentile(img[valid], [pmin, pmax])
    else:
        lo, hi = np.percentile(img, [pmin, pmax])

    if hi <= lo:
        hi = lo + 1.0

    img = np.clip((img - lo) / (hi - lo), 0, 1)
    img = exposure.adjust_gamma(img, gamma=gamma)

    if clahe:
        img = exposure.equalize_adapthist(img, clip_limit=0.02)

    return img


# =========================================================
# MAPAS FINALES
# =========================================================

DoP_map = np.sqrt(S1_map**2 + S2_map**2 + S3_map**2)
DoP_map = np.clip(np.nan_to_num(DoP_map, nan=0.0, posinf=0.0, neginf=0.0), 0, 1)

# Fondo más visible
I_disp = preparar_fondo(I_map, pmin=1, pmax=99, gamma=0.75, clahe=True)

# Para que el color solo aparezca donde realmente hay señal polarizada
mask_pol = DoP_map > 0.15
az_show = np.where(mask_pol, azimuth_deg, np.nan)
el_show = np.where(mask_pol, ellipticity_deg, np.nan)

# Transparencia guiada por DoP
alpha_pol = np.clip((DoP_map - 0.15) / 0.85, 0, 0.80)

# =========================================================
# FIGURA FINAL: estilo polarimetría de muestra biológica
# =========================================================

fig, axs = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

# --- DoP sobre intensidad ---
axs[0].imshow(I_disp, cmap='gray', interpolation='nearest')
im0 = axs[0].imshow(
    DoP_map,
    cmap='turbo',
    vmin=0,
    vmax=1,
    alpha=alpha_pol,
    interpolation='nearest'
)
axs[0].set_title("DoP")
axs[0].axis("off")
fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04, label="DoP")

# --- Azimut sobre intensidad ---
axs[1].imshow(I_disp, cmap='gray', interpolation='nearest')
im1 = axs[1].imshow(
    az_show,
    cmap='twilight_shifted',
    norm=TwoSlopeNorm(vmin=-90, vcenter=0, vmax=90),
    alpha=alpha_pol,
    interpolation='nearest'
)
axs[1].set_title("Azimut")
axs[1].axis("off")
fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04, label="Azimut (°)")

# --- Elipticidad sobre intensidad ---
axs[2].imshow(I_disp, cmap='gray', interpolation='nearest')
im2 = axs[2].imshow(
    el_show,
    cmap='seismic',
    norm=TwoSlopeNorm(vmin=-45, vcenter=0, vmax=45),
    alpha=alpha_pol,
    interpolation='nearest'
)
axs[2].set_title("Elipticidad")
axs[2].axis("off")
fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04, label="Elipticidad (°)")

plt.show()