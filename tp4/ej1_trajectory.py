import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2

def rosenbrock_gradient(x, y, a=1, b=100):
    df_dx = -2 * (a - x) - 4 * b * x * (y - x**2)
    df_dy = 2 * b * (y - x**2)
    return np.array([df_dx, df_dy])

# Algoritmo de gradiente descendente
def gradient_descent(start, learning_rate, tol=1e-6, max_iter=100000):
    x, y = start
    trajectory = [(x, y)]
    for i in tqdm(range(max_iter)):
        grad = rosenbrock_gradient(x, y)
        x, y = x - learning_rate * grad[0], y - learning_rate * grad[1]
        trajectory.append((x, y))
        if np.linalg.norm(grad) < tol:
            break
    return np.array(trajectory), rosenbrock(x, y), i + 1

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
gd_trajectory, _, _ = gradient_descent(start=(1, 0), learning_rate=0.0001)

# Llamar al gráfico 3D
plot_3d_function([gd_trajectory], ["Gradiente Descendente"])