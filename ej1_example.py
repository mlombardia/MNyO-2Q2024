import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definición de la función de Rosenbrock y su gradiente
def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2

def rosenbrock_gradient(x, y, a=1, b=100):
    df_dx = -2 * (a - x) - 4 * b * x * (y - x**2)
    df_dy = 2 * b * (y - x**2)
    return np.array([df_dx, df_dy])

# Algoritmo de gradiente descendente
def gradient_descent(start, learning_rate, tol=1e-6, max_iter=10000):
    x, y = start
    trajectory = [(x, y)]
    for i in range(max_iter):
        grad = rosenbrock_gradient(x, y)
        x, y = x - learning_rate * grad[0], y - learning_rate * grad[1]
        trajectory.append((x, y))
        if np.linalg.norm(grad) < tol:
            break
    return np.array(trajectory), rosenbrock(x, y), i + 1

# Parámetros iniciales
learning_rates = [0.0001, 0.001, 0.01, 0.1]
initial_conditions = [(0, 0), (-1, 1), (2, 2), (-1, -1)]

# Gráfica de los resultados
plt.figure(figsize=(12, 10))
x_vals = np.linspace(-2, 2, 400)
y_vals = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = rosenbrock(X, Y)

for lr in learning_rates:
    for start in initial_conditions:
        trajectory, final_val, num_iter = gradient_descent(start, lr)
        plt.contour(X, Y, Z, levels=np.logspace(0, 3, 20), cmap="jet")
        plt.plot(trajectory[:, 0], trajectory[:, 1], label=f"LR={lr}, Start={start}, Iter={num_iter}")
        plt.scatter(trajectory[-1, 0], trajectory[-1, 1], marker='x', color='red')

plt.title("Gradiente Descendente en la Función de Rosenbrock")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
# ------------------------------------------------------------------------------------------------
#NEWTON
import numpy as np
import matplotlib.pyplot as plt


# Definición de la función de Rosenbrock, gradiente y Hessiana
def rosenbrock(x, y, a=1, b=100):
    return (a - x) ** 2 + b * (y - x ** 2) ** 2


def rosenbrock_gradient(x, y, a=1, b=100):
    df_dx = -2 * (a - x) - 4 * b * x * (y - x ** 2)
    df_dy = 2 * b * (y - x ** 2)
    return np.array([df_dx, df_dy])


def rosenbrock_hessian(x, y, a=1, b=100):
    d2f_dx2 = 2 - 4 * b * (y - 3 * x ** 2)
    d2f_dxdy = -4 * b * x
    d2f_dy2 = 2 * b
    return np.array([[d2f_dx2, d2f_dxdy],
                     [d2f_dxdy, d2f_dy2]])


# Implementación del Método de Newton
def newton_method(start, tol=1e-6, max_iter=100):
    x, y = start
    trajectory = [(x, y)]
    for i in range(max_iter):
        grad = rosenbrock_gradient(x, y)
        hessian = rosenbrock_hessian(x, y)
        try:
            # Invertir la matriz Hessiana
            hessian_inv = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            print("Hessiana no invertible en la iteración:", i)
            break
        step = hessian_inv @ grad
        x, y = x - step[0], y - step[1]
        trajectory.append((x, y))
        if np.linalg.norm(grad) < tol:
            break
    return np.array(trajectory), rosenbrock(x, y), i + 1


# Comparación con Gradiente Descendente
def compare_methods(start, learning_rate=0.001):
    # Gradiente descendente
    gd_trajectory, gd_min, gd_iter = gradient_descent(start, learning_rate)

    # Método de Newton
    newton_trajectory, newton_min, newton_iter = newton_method(start)

    print(f"Condiciones iniciales: {start}")
    print(f"Gradiente Descendente -> Iteraciones: {gd_iter}, Valor mínimo: {gd_min}")
    print(f"Método de Newton -> Iteraciones: {newton_iter}, Valor mínimo: {newton_min}")

    # Gráficas
    plt.figure(figsize=(12, 6))
    x_vals = np.linspace(-2, 2, 400)
    y_vals = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = rosenbrock(X, Y)
    plt.contour(X, Y, Z, levels=np.logspace(0, 3, 20), cmap="jet")

    plt.plot(gd_trajectory[:, 0], gd_trajectory[:, 1], label="Gradiente Descendente", color="blue")
    plt.plot(newton_trajectory[:, 0], newton_trajectory[:, 1], label="Método de Newton", color="green")

    plt.scatter(gd_trajectory[-1, 0], gd_trajectory[-1, 1], marker='x', color='blue', label="Final GD")
    plt.scatter(newton_trajectory[-1, 0], newton_trajectory[-1, 1], marker='x', color='green', label="Final Newton")

    plt.title("Comparación: Gradiente Descendente vs Método de Newton")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


# Probar métodos con condiciones iniciales
compare_methods(start=(0, 0))


# ------------------------------------------------------------------------------------------------
# PLOT 3D
def plot_3d_function(trajectories, labels):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Superficie de la función
    x_vals = np.linspace(-2, 2, 400)
    y_vals = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = rosenbrock(X, Y)
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, edgecolor='none')

    # Agregar trayectorias de los métodos
    for trajectory, label in zip(trajectories, labels):
        x, y = trajectory[:, 0], trajectory[:, 1]
        z = [rosenbrock(px, py) for px, py in trajectory]
        ax.plot(x, y, z, label=label, linewidth=2)
        ax.scatter(x[-1], y[-1], z[-1], color='red')  # Punto final

    ax.set_title("Visualización 3D de la función de Rosenbrock")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    ax.legend()
    plt.show()

# Generar las trayectorias
gd_trajectory, _, _ = gradient_descent(start=(0, 0), learning_rate=0.001)
newton_trajectory, _, _ = newton_method(start=(0, 0))

# Llamar al gráfico 3D
plot_3d_function([gd_trajectory, newton_trajectory], ["Gradiente Descendente", "Método de Newton"])
