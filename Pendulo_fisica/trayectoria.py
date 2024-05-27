import numpy as np
import matplotlib.pyplot as plt

# Parámetros
theta_0 = 0.2  # Ángulo inicial en radianes (aproximadamente 11.5 grados)
beta = 0.01    # Coeficiente de amortiguamiento
omega_d = np.sqrt(9.81/1)  # Frecuencia angular amortiguada para un péndulo de 1m de longitud
delta = 0.0    # Ángulo de fase

# Datos sintéticos
t = np.linspace(0, 20, 500)  # Generar datos de tiempo de 0 a 20 segundos
theta_t = theta_0 * np.exp(-beta * t) * np.cos(omega_d * t + delta)

# Ruido para simular mediciones reales
ruido = np.random.normal(0, 0.005, t.shape)
theta_con_ruido = theta_t + ruido

# Gráfico
plt.plot(t, theta_con_ruido, label='Ángulo Medido')
plt.plot(t, theta_t, 'k--', label='Ángulo Verdadero')
plt.xlabel('Tiempo (s)')
plt.ylabel('Ángulo (rad)')
plt.legend()
plt.show()