import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


# Definición de la función de Rosenbrock y su gradiente
def rosenbrock(x, y, a=1, b=100):
    return (a - x) ** 2 + b * (y - x ** 2) ** 2


def rosenbrock_gradient(x, y, a=1, b=100):
    df_dx = -2 * (a - x) - 4 * b * x * (y - x ** 2)
    df_dy = 2 * b * (y - x ** 2)
    return np.array([df_dx, df_dy])


# Algoritmo de gradiente descendente
def gradient_descent(start, learning_rate, tol=1e-7, max_iter=500000):
    x, y = start
    trajectory = [(x, y)]
    gradient_norms = []
    for i in range(max_iter):
        grad = rosenbrock_gradient(x, y)
        gradient_norms.append(np.linalg.norm(grad))
        x, y = x - learning_rate * grad[0], y - learning_rate * grad[1]
        trajectory.append((x, y))
        if np.linalg.norm(grad) < tol:
            break
    return np.array(trajectory), rosenbrock(x, y), i + 1, gradient_norms


# Parámetros iniciales
learning_rates = [0.0001, 0.001, 0.01, 0.1]
# learning_rates = [0.01, 0.1]
initial_conditions = [(0, 0), (-1, 1), (2, 2), (-1, -1)]

# Tabular resultados
results = []

# Gráfica de los resultados
plt.figure(figsize=(12, 10))
x_vals = np.linspace(-2, 2, 400)
y_vals = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = rosenbrock(X, Y)



plt.figure(figsize=(12, 6))

# Iterar sobre las tasas de aprendizaje y condiciones iniciales
for lr in learning_rates:
    for start in initial_conditions:
        trajectory, final_val, num_iter, gradient_norms = gradient_descent(start, lr)
        results.append({
            "Tasa de Aprendizaje": lr,
            "Condición Inicial": start,
            "Valor Mínimo": final_val,
            "Iteraciones": num_iter,
            "Convergencia": "Sí" if np.linalg.norm(rosenbrock_gradient(*trajectory[-1])) < 1e-7 else "No"
        })
        plt.plot(range(len(gradient_norms)), gradient_norms, label=f"LR={lr}, Start={start}")

# Configuración del gráfico
plt.yscale("log")  # Escala logarítmica en el eje y
plt.xscale("log")  # También podríamos aplicar escala logarítmica en x para iteraciones muy grandes
plt.grid(which="both", linestyle="--", linewidth=0.5)  # Rejilla para mayor claridad
plt.title("Convergencia del Gradiente (||∇f||) - Escala Logarítmica")
plt.xlabel("Iteraciones (log)")
plt.ylabel("||∇f|| (log)")
plt.legend(fontsize="small", loc="upper right")
plt.show()

# Tabular resultados
df_results = pd.DataFrame(results)
print("Resultados Tabulados:")
print(df_results)
