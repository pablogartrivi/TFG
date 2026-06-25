import rawpy
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================
# CONFIG
# ============================================
#Carpeta ocn imagenes dark y light
light_folder = ""
#Carpeta para guardar los resultados
output_folder = ""
os.makedirs(output_folder, exist_ok=True)

all_files = os.listdir(light_folder)

light_files = []
dark_files = {}

for file in all_files:

    if not file.endswith(".dng"):
        continue

    if file.endswith("_d.dng"):
        key = file.replace("_d.dng", "")
        dark_files[key] = file
    else:
        key = file.replace(".dng", "")
        light_files.append((key, file))

# ============================================
# FUNCIONES
# ============================================

def parse_exposure(f):
    f = f.replace(",", ".").strip()
    if "-" in f:
        a, b = f.split("-")
        return float(a) / float(b)
    return float(f)

"""
La función lee los valores RGB, devolviendo al amtriz de valores de intensidad por cada pixel
Raw_white_level es el nivel máximo que puede leer un bit, y por tanto el nivel de saturación
"""


def read_raw(path):
    with rawpy.imread(path) as raw:
        return raw.raw_image_visible.astype(np.float64), raw.white_level


"""
Función de calculo de la SNR, le llega la imagen normal y la imagen dark
signal = intensidad media, represenan los fotones detectados
Calculamos la varianza del dark frame para tener el nivel de ruido del sensor (ruido termico, ruido electrico, ADC)
Generamos un modelo de ruido de possion para modelar mejor el ruido de cada señal
Se suman los ruidos y el resultado se divide a la señal útil

"""


def compute_snr_with_dark(signal_img, dark_img):
    """
    SNR más realista usando dark frame
    """
    signal = np.mean(signal_img)

    # ruido del dark (read noise + thermal)
    read_noise = np.std(dark_img)

    # ruido de Poisson
    #poisson_noise = np.sqrt(np.abs(signal))
    poisson_noise = np.sqrt(np.maximum(signal, 1))

    total_noise = np.sqrt(poisson_noise**2 + read_noise**2)

    if total_noise == 0:
        return 0

    return signal / total_noise


def extract_exposure_ns(name):
    try:
        # EXP_100000_ns_ISO21
        return float(name.split("_")[1])
    except:
        return None

def ns_to_seconds(ns):
    return ns / 1e9

# ============================================
# CARGA DATOS
# ============================================

times, I_list, sat_flags, names = [], [], [], []
snr_list = []

#start_index = file_names.index(START_FROM)

for key, light_file in light_files:

    if key not in dark_files:
        print(f"Falta dark frame: {key}")
        continue

    light_path = os.path.join(light_folder, light_file)
    dark_path = os.path.join(light_folder, dark_files[key])

    # EXTRAER TIEMPO
    exposure_ns = extract_exposure_ns(key)
    if exposure_ns is None:
        print(f"No se pudo leer exposición: {key}")
        continue

    t = ns_to_seconds(exposure_ns)

        # Leer imágenes
    light_img, SATURATION_LEVEL = read_raw(light_path)      #señal + nivel sat
    dark_img, _ = read_raw(dark_path)                       #dark_frame
    # eliminar offset del dark
    dark_mean = np.mean(dark_img)                           #Se calcula el offset de la camara
    dark_img = dark_img - dark_mean      
    img = light_img - dark_img        
    # ========================================
    # LIMPIEZA
    # ========================================
    p99 = np.percentile(img, 99.9)
    img = np.clip(img, 0, p99)                              #Eliminamos posibl reuido residual que haya quedado en el percentil 99.9 

    # canal green
    #green = img[1::2, 1::2]                                   #Selección ROI
    h, w = img.shape
    roi = img[h//4:3*h//4, w//4:3*w//4]
    red = roi[1::2, 0::2]
    mean_I = np.mean(red)

    # saturación
    max_pixel = np.max(img)                                        #Detecta si el sensor esta saturado y lo guarda en caso de que este
    sat = np.percentile(img, 99.9) >= SATURATION_LEVEL                  

    # SNR con dark
    #snr = compute_snr_with_dark(green, dark_img[1::2, 1::2])
    dark_roi = dark_img[h//4:3*h//4, w//4:3*w//4]                   #Realizamos el mismo proceso con la ROI del df
    dark_red = dark_roi[1::2, 0::2]
    snr = compute_snr_with_dark(red, dark_red)                 #Calculamos snr 

    #t = parse_exposure(f)

    print(f"{key} -> t={t:.6f}s| mean={mean_I:.2f} | SNR={snr:.2f} | sat={sat}")     #Escribimos resultados por pantalla

    times.append(t)
    I_list.append(mean_I)
    sat_flags.append(sat)
    names.append(key)
    snr_list.append(snr)

# ============================================
# CONVERSIÓN Y ORDEN
# ============================================

times = np.array(times)                 #Convetimos arrays en num py para tratado de datos más sencillo
I_list = np.array(I_list)
sat_flags = np.array(sat_flags, dtype=bool)
names = np.array(names)
snr_list = np.array(snr_list)

idx = np.argsort(times)             #Ordenamos por tiempo

times = times[idx]                      #Reordenación de todos los datos pro tiempos 
I_list = I_list[idx]
sat_flags = sat_flags[idx]
names = names[idx]
snr_list = snr_list[idx]

# ============================================
# FILTRO NO SATURADAS
# ============================================

mask = ~sat_flags

t = times[mask]
I = I_list[mask]
names_valid = names[mask]           #Elegimos unicmaente imageens no saturadas

# eliminar zona ruido
#noise_floor = np.median(I_list[:5])
#mask_signal = I > noise_floor * 1.5

snr_valid = snr_list[mask]

mask_signal = snr_valid > 0

t = t[mask_signal]
I = I[mask_signal]
names_valid = names_valid[mask_signal]      #Eliminamos todos los valores saturados

print("\nDEBUG")
print("len(times) =", len(times))
print("len(t) =", len(t))
print("len(I) =", len(I))
print("len(snr_valid) =", len(snr_valid))

print("snr_valid =", snr_valid)
print("t =", t)
print("I =", I)

# ============================================
# BÚSQUEDA ZONA LINEAL
# ============================================

R2_threshold = 0.99
best_length = 0
best_i, best_j = 0, 0
best_R2 = 0
min_points = 5

for i in range(len(t)):
    for j in range(i + min_points, len(t) + 1):

        tt = t[i:j]
        II = I[i:j]

        if len(tt) < min_points:
            continue

        coeffs_temp = np.polyfit(tt, II, 1)
        fit_temp = np.polyval(coeffs_temp, tt)

        SS_res = np.sum((II - fit_temp)**2)
        SS_tot = np.sum((II - np.mean(II))**2)

        R2 = 1 - SS_res / SS_tot

        if R2 > R2_threshold:
            length = j - i
            if length > best_length:
                best_length = length
                best_i, best_j = i, j
                best_R2 = R2

# ============================================
# RESULTADOS LINEALES
# ============================================

t_lin = t[best_i:best_j]
I_lin = I[best_i:best_j]
names_lin = names_valid[best_i:best_j]

coeffs = np.polyfit(t_lin, I_lin, 1)
fit = np.polyval(coeffs, t_lin)

# ============================================
# SNR DE LA ZONA LINEAL
# ============================================

snr_valid = snr_list[mask]
snr_valid = snr_valid[mask_signal]

snr_lin_values = snr_valid[best_i:best_j]

# ============================================
# GRÁFICOS
# ============================================

# Ajuste para toda la zona válida
fit_all_valid = np.polyval(coeffs, t)

# Residuo en la zona lineal
residuals_lin = I_lin - fit
residuals_rel_lin = residuals_lin / fit * 100

# Crear figura 1: Intensidad vs tiempo + ajuste
plt.figure(figsize=(10, 6))
plt.plot(times, I_list, "o", ms=10, alpha=0.35, label="Todos los puntos")
plt.plot(t_lin, I_lin, "o", ms=6, label="Zona lineal")
plt.plot(t_lin, fit, "-", lw=2, label=f"Ajuste lineal (R²={best_R2:.4f})")
plt.xlabel("Tiempo de exposición (s)")
plt.ylabel("Intensidad media (ADU)")
plt.title("Linealidad del sensor: intensidad media vs tiempo")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "01_linealidad_intensidad_vs_tiempo.png"), dpi=300)
plt.close()

# Figura 2: Residuos absolutos y relativos
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax[0].axhline(0, color="k", lw=1)
ax[0].plot(t_lin, residuals_lin, "o-", ms=5)
ax[0].set_ylabel("Residuo (ADU)")
ax[0].set_title("Residuos del ajuste lineal")
ax[0].grid(True, alpha=0.3)

ax[1].axhline(0, color="k", lw=1)
ax[1].plot(t_lin, residuals_rel_lin, "o-", ms=5)
ax[1].set_xlabel("Tiempo de exposición (s)")
ax[1].set_ylabel("Residuo relativo (%)")
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "02_residuos_linealidad.png"), dpi=300)
plt.close()

# Figura 3: SNR vs tiempo
plt.figure(figsize=(10, 6))
plt.plot(times, snr_list, "o", ms=4, alpha=0.35, label="Todos")
plt.plot(t, snr_valid, "o", ms=5, label="Válidos")
plt.plot(t_lin, snr_lin_values, "o", ms=6, label="Zona lineal")
plt.xlabel("Tiempo de exposición (s)")
plt.ylabel("SNR")
plt.title("SNR vs tiempo de exposición")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "03_snr_vs_tiempo.png"), dpi=300)
plt.close()

# Figura 4: Pendiente local
if len(t_lin) > 2:
    dI = np.diff(I_lin)
    dt = np.diff(t_lin)
    local_slope = dI / dt
    t_mid = (t_lin[:-1] + t_lin[1:]) / 2

    plt.figure(figsize=(10, 6))
    plt.plot(t_mid, local_slope, "o-", ms=5)
    plt.axhline(coeffs[0], color="r", ls="--", lw=2, label="Pendiente global")
    plt.xlabel("Tiempo de exposición (s)")
    plt.ylabel("Pendiente local dI/dt")
    plt.title("Variación de la pendiente en la zona lineal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "04_pendiente_local.png"), dpi=300)
    plt.close()

# Error relativo frente a intensidad
plt.figure(figsize=(10, 6))
plt.axhline(0, color="k", lw=1)
plt.plot(I_lin, residuals_rel_lin, "o-", ms=5)
plt.xlabel("Intensidad medida (ADU)")
plt.ylabel("Error relativo (%)")
plt.title("Error relativo de linealidad vs intensidad")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "05_error_relativo_vs_intensidad.png"), dpi=300)
plt.close()

# ============================================
# GUARDADO
# ============================================

with open(os.path.join(output_folder, "resultados.txt"), "w", encoding="utf-8") as f:

    f.write("\n" + "="*60 + "\n")
    f.write("     ANALISIS FOTOMETRICO (CON DARK FRAMES)\n")
    f.write("="*60 + "\n\n")

    # ============================================
    # RESUMEN
    # ============================================

    f.write(" RESUMEN GENERAL\n")
    f.write("-"*40 + "\n")
    f.write(f"Imagenes analizadas        : {len(times)}\n")
    f.write(f"Imagenes validas           : {len(t)}\n")
    f.write(f"Imagenes zona lineal       : {len(t_lin)}\n")
    f.write(f"Rango lineal (imagenes)    : {names_lin[0]} → {names_lin[-1]}\n")
    f.write(f"Rango lineal (tiempo)      : {t_lin[0]:.6e}s → {t_lin[-1]:.6e}s\n\n")

    # ============================================
    # MODELO LINEAL
    # ============================================

    f.write(" MODELO LINEAL\n")
    f.write("-"*40 + "\n")
    f.write("I = k · t + b\n")
    f.write(f"k (ganancia)              : {coeffs[0]:.6f}\n")
    f.write(f"b (offset residual)       : {coeffs[1]:.6f}\n")
    f.write(f"R² final                  : {best_R2:.6f}\n\n")

    # ============================================
    # ESTADÍSTICOS ZONA LINEAL
    # ============================================

    mean_lin = np.mean(I_lin)
    std_lin = np.std(I_lin)
    var_lin = np.var(I_lin)

    f.write(" ESTADISTICOS (ZONA LINEAL)\n")
    f.write("-"*40 + "\n")
    f.write(f"Media intensidad          : {mean_lin:.3f}\n")
    f.write(f"Desviacion std            : {std_lin:.3f}\n")
    f.write(f"Varianza                  : {var_lin:.3f}\n")
    f.write(f"SNR (media/std)           : {mean_lin/std_lin:.3f}\n\n")

    # ============================================
    # SNR REAL (con dark)
    # ============================================

    # aplicar mismos filtros que a t e I
    snr_valid = snr_list[mask]
    snr_valid = snr_valid[mask_signal]

    
    snr_lin_values = snr_valid[best_i:best_j]

    f.write(" SNR (CON DARK)\n")
    f.write("-"*40 + "\n")
    f.write(f"SNR medio zona lineal     : {np.mean(snr_lin_values):.3f}\n")
    f.write(f"SNR std                   : {np.std(snr_lin_values):.3f}\n")
    f.write(f"SNR minimo                : {np.min(snr_lin_values):.3f}\n")
    f.write(f"SNR maximo                : {np.max(snr_lin_values):.3f}\n\n")


    # ============================================
    # PARAMETROS AVANZADOS
    # ============================================

    # rango dinámico en zona lineal
    I_min = np.min(I_lin)
    I_max = np.max(I_lin)
    # ruido mínimo (dark frame)
    dark_noise = np.std(dark_red)

    # señal máxima (antes de saturación)
    I_sat = np.max(I_list[~sat_flags])  # máximo no saturado



    dynamic_range_real = 20 * np.log10(I_sat / dark_noise)


    # offset relativo
    offset_ratio = coeffs[1] / np.mean(I_lin)

    # estabilidad de la pendiente (linealidad diferencial)
    dI = np.diff(I_lin)
    dt = np.diff(t_lin)
    local_slope = dI / dt
    slope_variation = np.std(local_slope) / np.mean(local_slope) * 100

    f.write(" PARAMETROS AVANZADOS\n")
    f.write("-"*40 + "\n")
    f.write(f"Rango dinamico (dB)        : {dynamic_range_real:.2f}\n")
    f.write(f"Offset relativo            : {offset_ratio:.6f}\n")
    f.write(f"Variacion pendiente (%)    : {slope_variation:.3f}\n\n")

    # ============================================
    # LINEALIDAD
    # ============================================

    deviation = I_lin - fit
    linearity_error = np.max(np.abs(deviation / fit)) * 100
    linearity_rms = np.sqrt(np.mean((deviation / fit)**2)) * 100
    relative_error = np.abs(deviation / I_lin)
    rel_error_max = np.max(relative_error) * 100
    rel_error_rms = np.sqrt(np.mean(relative_error**2)) * 100

    f.write(" LINEALIDAD DEL SENSOR\n")
    f.write("-"*40 + "\n")
    f.write(f"Error relativo max (%)     : {rel_error_max:.3f}\n")
    f.write(f"Error relativo RMS (%)     : {rel_error_rms:.3f}\n\n")

    # ============================================
    # LISTA DE IMÁGENES
    # ============================================

    f.write(" IMAGENES EN ZONA LINEAL\n")
    f.write("-"*40 + "\n")

    for i in range(len(names_lin)):
        f.write(
            f"{names_lin[i]:>8}  |  "
            f"t={t_lin[i]:.6e}s  |  "
            f"I={I_lin[i]:.2f}  |  "
            f"SNR={snr_lin_values[i]:.2f}\n"
        )

    f.write("\n" + "="*60 + "\n")
    f.write("FIN DEL ANALISIS\n")
    f.write("="*60 + "\n")

print("\n OK -> análisis con DARK completado")