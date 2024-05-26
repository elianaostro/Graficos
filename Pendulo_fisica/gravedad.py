
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

longitudes = [0.5, 0.6, 0.7, 0.8, 0.9]  # Ejemplo de longitudes en metros
periodos = [1.42, 1.55, 1.67, 1.79, 1.89]  # Ejemplo de periodos en segundos para las longitudes anteriores

# Definimos la función lineal para el ajuste
def linear_func(l, m, c):
    return m * l + c

# Cuadrado de los periodos
T_squared = np.square(periodos)

# Ajuste lineal
params, covariance = curve_fit(linear_func, longitudes, T_squared)
m, c = params

# Valores ajustados
T_squared_fit = linear_func(np.array(longitudes), m, c)

# Estimar g a partir de la pendiente
g_estimado = 4 * np.pi**2 / m

# Graficar
plt.figure(figsize=(10, 6))
plt.scatter(longitudes, T_squared, label='Datos', color='blue')
plt.plot(longitudes, T_squared_fit, label=f'Ajuste Lineal: 4π²/g = {m:.2f}', color='red')
plt.xlabel('Longitud (m)')
plt.ylabel(' (s^2)')
plt.title(f'Estimación de g: {g_estimado:.2f} m/s^2')
plt.legend()
plt.grid(True)
plt.show()
