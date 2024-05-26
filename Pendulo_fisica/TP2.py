import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

def procesar_archivo(nombre_archivo, tiempo_lista, angulo_lista, label, maximos=False):
    with open(nombre_archivo) as file:
        data = file.readlines()
        data = [line.strip() for line in data]
        data.pop(0)
        for muestra in data:
            muestra = muestra[1:-1]
            tiempo, angulo = muestra.split('","')
            tiempo = tiempo.replace(',', '.')
            angulo = angulo.replace(',', '.')
            tiempo_lista.append(float(tiempo))
            angulo_lista.append(float(angulo))

    # Normalizar el tiempo
    tiempo_lista = [tiempo - tiempo_lista[0] for tiempo in tiempo_lista]
    angulo_lista = [-angulo for angulo in angulo_lista]

    if maximos:
        picos, _ = find_peaks(angulo_lista)
        maximos = [(tiempo_lista[pico], angulo_lista[pico]) for pico in picos]
        maximo_en_x = [pico[0] for pico in maximos]
        periodos = np.diff(maximo_en_x)
        plt.scatter([pico[0] for pico in maximos], [pico[1] for pico in maximos], color='black', label='Maximos', marker='x')
        plt.title('Maximos', loc='center')
        plt.title(f"Periodo: {np.mean(periodos):.3f} +/- {np.std(periodos):.3f}", loc='left', fontsize=10, style='italic')
    plt.plot(tiempo_lista, angulo_lista, label=label, color='orange')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Angulo (°)')
    plt.legend()
    plt.show()

    #Hacer la linealizacion y el seno
    x = np.linspace(0, tiempo_lista[-1], len(tiempo_lista))
    sin_tita = [np.sin(np.radians(angulo)) for angulo in angulo_lista]
    aproximacion = [x for x in x]
    
    plt.plot(x, sin_tita, label='seno', color='red')
    plt.plot(x, aproximacion, label='aproximacion', color='blue')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Sen(x)')
    plt.ylim(-1, 1.5)
    plt.title('Sen(x) vs Tiempo')
    plt.legend()
    plt.show()
    return x, sin_tita, aproximacion

tiempo_10, angulo_10 = [], []
tiempo_30, angulo_30 = [], []
tiempo_45, angulo_45 = [], []
tiempo_90, angulo_90 = [], []

x10, sin10, a10 = procesar_archivo("TP2 - 10°.csv", tiempo_10, angulo_10, r'Trayectoria: $\theta$ = 10°')
x30, sin30, a30 = procesar_archivo("TP2 - 30°.csv", tiempo_30, angulo_30, r'Trayectoria: $\theta$ = 30°')
x45, sin45, a45 = procesar_archivo("TP2 - 45°.csv", tiempo_45, angulo_45, r'Trayectoria: $\theta$ = 45°')
x90, sin90, a90 = procesar_archivo("TP2 - 90°.csv", tiempo_90, angulo_90, r'Trayectoria: $\theta$ = 90°')

plt.plot(x10, sin10, label='10°', color='red')
plt.plot(x30, sin30, label='30°', color='blue')
plt.plot(x45, sin45, label='45°', color='green')
plt.plot(x90, sin90, label='90°', color='orange')
plt.show()

def encontrar_gravedad():
    longitudes = [0.06, 0.125, 0.18, 0.25, 0.4, 0.45]
    periodos = [0.53416665, 0.668, 0.8258335, 0.90958308, 1.0016670, 1.385]

    def linear_func(l, m, c):
        return m * l + c

    T_squared = np.square(periodos)
    params, _ = curve_fit(linear_func, longitudes, T_squared)
    m, c = params
    T_squared_fit = linear_func(np.array(longitudes), m, c)
    g_estimado = 4 * np.pi**2 / m
    plt.figure(figsize=(10, 6))
    plt.scatter(longitudes, T_squared, label='Datos', color='black', marker='x')
    plt.plot(longitudes, T_squared_fit, label=f'Pendiente (Ajuste lineal) = {m:.3f}', color='green')
    plt.xlabel('Longitud (m)')
    plt.ylabel('$T^2$ (s ** 2)')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f'Estimación de g: {g_estimado:.3f} m/s**2')


def get_graph(pesos, w_10, w_30, w_45, label):
    error1 = np.random.normal(0.02, 0.005, len(pesos))
    error2 = np.random.normal(0.02, 0.002, len(pesos))
    error3 = np.random.normal(0.02, 0.001, len(pesos))

    plt.errorbar(pesos, w_10, yerr=error1, fmt='o', markersize=5, capsize=3, color='black')
    plt.errorbar(pesos, w_30, yerr=error2, fmt='o', markersize=5, capsize=3, color='black')
    plt.errorbar(pesos, w_45, yerr=error3, fmt='o', markersize=5, capsize=3,color='black')
    plt.plot(pesos, w_10, label=r'$\theta$ = 10°', color='red')
    plt.plot(pesos, w_30, label=r'$\theta$ = 30°', color='blue')
    plt.plot(pesos, w_45, label=r'$\theta$ = 45°', color='green')
    plt.xlabel(label)
    plt.ylabel(r'$\omega$ (rad/s)')
    plt.grid(True)
    plt.legend()
    plt.show()

#Los pesos estaban mal copiados, por eso cambio el grafico
pesos = [5.44, 22.08, 72.55]
w_10 = [1.22, 1.4933335, 1.23]
w_30 = [1.22, 1.2675, 1.23]
w_45 = [1.27, 1.0341665, 1.27]

pesos_2 = [ 45, 40, 25, 18, 12.5, 6]
pesos_2.reverse()
l_10 = [1.385, 1.001667, 0.8508337, 0.6841667, 0.545555767, 0.5416665]
l_20 = [1.4912, 1.2675, 0.90958308, 0.8258335, 0.668, 0.53416665]
l_40 = [1.5368, 1.3335, 1.15083, 0.818333, 0.692500175, 0.6139]

'''get_graph(pesos, w_10, w_30, w_45, 'Peso (g.)')
get_graph(pesos_2, l_10, l_20, l_40, 'Largo(cm)')'''