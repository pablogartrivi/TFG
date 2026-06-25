import os
import re
import rawpy
import numpy as np

# ==============================
# CONFIGURACIÓN
# ==============================
ruta_imagenes = ""
ruta_dark = ""
output_file = "resultados_snr.txt"

# ==============================
# FUNCIONES
# ==============================



from scipy.ndimage import uniform_filter

def extraer_roi_max_intensidad(img, size=200):
    """
    ROI basada en máxima intensidad media local (MUCHO más estable que argmax).
    """

    # media local en ventana size x size
    local_mean = uniform_filter(img, size=size)

    # localizar máximo de esa media local
    y_max, x_max = np.unravel_index(np.argmax(local_mean), local_mean.shape)

    h, w = img.shape
    half = size // 2

    y1 = max(0, y_max - half)
    y2 = min(h, y_max + half)
    x1 = max(0, x_max - half)
    x2 = min(w, x_max + half)

    return img[y1:y2, x1:x2]

def extraer_exposicion(nombre):
    match = re.search(r'EXP_(\d+)', nombre)
    return int(match.group(1)) if match else None


def cargar_raw(path):
    with rawpy.imread(path) as raw:
        img = raw.raw_image_visible.astype(np.float32)
        white_level = raw.white_level
    return img, white_level


def agrupar_por_exposicion(ruta):
    grupos = {}
    for file in os.listdir(ruta):
        if file.endswith(".dng"):
            exp = extraer_exposicion(file)
            if exp is not None:
                grupos.setdefault(exp, []).append(os.path.join(ruta, file))
    return grupos


from scipy.ndimage import uniform_filter
import numpy as np

def calcular_snr(signal_img, dark_img, white_level, size=250):

    # ROI basada en máxima relación señal/fondo
    local_signal = uniform_filter(signal_img, size=size)
    local_dark = uniform_filter(dark_img, size=size)

    local_metric = local_signal / (local_dark + 1e-12)

    y_max, x_max = np.unravel_index(np.argmax(local_metric),
                                    local_metric.shape)

    h, w = signal_img.shape
    half = size // 2

    y1 = max(0, y_max - half)
    y2 = min(h, y_max + half)
    x1 = max(0, x_max - half)
    x2 = min(w, x_max + half)

    signal_roi = signal_img[y1:y2, x1:x2]
    dark_roi = dark_img[y1:y2, x1:x2]

    min_h = min(signal_roi.shape[0], dark_roi.shape[0])
    min_w = min(signal_roi.shape[1], dark_roi.shape[1])

    signal_roi = signal_roi[:min_h, :min_w]
    dark_roi = dark_roi[:min_h, :min_w]

    # eliminar píxeles saturados
    mask = signal_roi < white_level * 0.999

    signal = signal_roi[mask]
    dark = dark_roi[mask]

    if signal.size == 0:
        return np.nan

    # señal corregida
    signal_corr = signal - dark

    # μ_S
    mu_s = np.mean(signal_corr)

    # σ_dark
    sigma_dark = np.std(dark)

    # SNR física
    snr = mu_s / np.sqrt(mu_s + sigma_dark**2)

    return snr
# ==============================
# PROCESAMIENTO
# ==============================

grupos_signal = agrupar_por_exposicion(ruta_imagenes)
grupos_dark = agrupar_por_exposicion(ruta_dark)

resultados = []

for exp in grupos_signal:

    if exp not in grupos_dark:
        print(f"No hay dark frame para EXP_{exp}")
        continue

    for img_path in grupos_signal[exp]:

        dark_path = grupos_dark[exp][0]

        img, white_level = cargar_raw(img_path)
        mask = img < white_level
        img_valid = img[mask]
        dark, _ = cargar_raw(dark_path)

        # Métricas
        mean_intensity = np.mean(img_valid)
        std_intensity = np.std(img_valid)

        saturated_pixels = np.sum(img >= white_level)
        total_pixels = img.size
        saturation_ratio = saturated_pixels / total_pixels

        snr = calcular_snr(img, dark, white_level)
        snr = 20*np.log10(snr)
        # Guardar en lista
        resultados.append({
            "imagen": os.path.basename(img_path),
            "exp": exp,
            "mean": mean_intensity,
            "std": std_intensity,
            "sat_pixels": saturated_pixels,
            "sat_ratio": saturation_ratio,
            "snr": snr
        })


# ==============================
# ORDENAR RESULTADOS
# ==============================

resultados_ordenados = sorted(resultados, key=lambda x: x["exp"])

# ==============================
# GUARDAR EN TXT
# ==============================

with open(output_file, "w") as f:

    exp_actual = None

    for r in resultados_ordenados:

        # Separador por exposición 
        if r["exp"] != exp_actual:
            f.write(f"\n===== EXPOSICIÓN: {r['exp']} =====\n")
            exp_actual = r["exp"]

        f.write(f"Imagen: {r['imagen']}\n")
        f.write(f"Media intensidad: {r['mean']:.2f}\n")
        f.write(f"STD intensidad: {r['std']:.2f}\n")
        f.write(f"Pixeles saturados: {r['sat_pixels']}\n")
        f.write(f"Ratio saturacion: {r['sat_ratio']:.6f}\n")
        f.write(f"SNR: {r['snr']:.2f}\n")
        f.write("-" * 40 + "\n")
print("Guardado en:", os.path.abspath(output_file))
print("Procesamiento terminado (ordenado por exposición).")