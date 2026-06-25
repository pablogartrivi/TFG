import rawpy
import numpy as np

# ============================================
# CARGA RAW
# ============================================

path = ""   

with rawpy.imread(path) as raw:
    raw_img = raw.raw_image_visible
    pattern = raw.raw_pattern

# ============================================
# MOSTRAR PATRÓN
# ============================================

print("Patrón Bayer (índices):")
print(pattern)

print("\nSignificado de índices:")
print("0 = Rojo (R)")
print("1 = Verde (G)")
print("2 = Azul (B)")

# ============================================
# TRADUCIR A FORMATO LEGIBLE
# ============================================

mapping = {
    0: "R",
    1: "G",
    2: "B",
    3: "G"   # MUY IMPORTANTE
}

pattern_str = np.vectorize(mapping.get)(pattern)

print("\nPatrón Bayer (formato 2x2):")
print(pattern_str)

# ============================================
# RESULTADO FINAL
# ============================================

pattern_name = "".join(pattern_str.flatten())
print(f"\nTu patrón Bayer es: {pattern_name}")