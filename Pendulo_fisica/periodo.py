from scipy.signal import find_peaks

# Encontrar picos positivos
picos, _ = find_peaks(theta_con_ruido, distance=omega*2, height=0.15)

# Extraer alturas de los picos
alturas_picos = theta_con_ruido[picos]

# Calcular el período de la oscilación
periodos = np.diff(t[picos])

print(f"El período es {np.mean(periodos):.1f} +/- {np.std(periodos):.1f}")

# Gráfica de Picos
plt.plot(t, theta_con_ruido)
plt.plot(t[picos], alturas_picos, "x")
plt.xlabel('Tiempo (s)')
plt.ylabel('Ángulo (rad)')
plt.title('Picos Identificados en la Oscilación')
plt.show()
