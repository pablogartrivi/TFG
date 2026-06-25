import numpy as np

# Imatrix de la ecuación (4.12) del trabajo
Imatrix = np.array([
])

# A obtenida en el trabajo (ec. 4.13)
A = np.array([
])

# Varianzas experimentales (modelo Poisson: sigma^2 ≈ I_media por columna)
I_mean_per_config = Imatrix.mean(axis=1)  # media por fila del PSA
Sigma_I = np.diag(I_mean_per_config)

# Covarianza del estimador OLS bajo ruido real heterocedástico
AtA_inv = np.linalg.inv(A.T @ A)
Cov_OLS = AtA_inv @ A.T @ Sigma_I @ A @ AtA_inv

# Covarianza del estimador WLS óptimo
W = np.diag(1.0 / I_mean_per_config)
AtWA_inv = np.linalg.inv(A.T @ W @ A)
Cov_WLS = AtWA_inv

# Ineficiencia relativa (traza normalizada)
ineff = (np.trace(Cov_OLS) - np.trace(Cov_WLS)) / np.trace(Cov_WLS) * 100
print(f"Ineficiencia relativa OLS vs WLS: {ineff:.1f}%")