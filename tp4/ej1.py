import numpy as np
import matplotlib.pyplot as plt

# -------------Implementar algoritmo gradient descent--------------
def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2

def rosenbrock_gradient(x, y, a=1, b=100):
    df_dx = -2 * (a - x) - 4 * b * x * (y - x**2)
    df_dy = 2 * b * (y - x**2)
    return np.array([df_dx, df_dy])

def gradient_descent(start, learning_rate, tol=0.5, max_iter=15000):
    print("start: ",start," lr: ", learning_rate)
    x, y = start
    trajectory = [(x, y)]
    gradient_norms = []
    for i in range(max_iter):
        grad = rosenbrock_gradient(x, y)
        gradient_norms.append(np.linalg.norm(grad))
        x, y = x - learning_rate * grad[0], y - learning_rate * grad[1]
        trajectory.append((x, y))
        print("-----------------")
        print(x, " ", y)
        print(grad)
        print("-----------------")
        if np.linalg.norm(grad) < tol:
            break
    return np.array(trajectory), rosenbrock(x, y), i + 1, gradient_norms

# -------------------Probar distintos eta----------------------------

learning_rates = [0.00001,0.0001,0.001]
initial_conditions = [(0,0)]

plt.figure(figsize=(12, 10))
x_vals = np.linspace(-2, 2, 400)
y_vals = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = rosenbrock(X, Y)

for lr in learning_rates:
    for start in initial_conditions:
        trajectory, final_val, num_iter, gradient_norms = gradient_descent(start, lr)
        plt.plot(range(len(gradient_norms)), gradient_norms, label=f"LR={lr}, Start={start}")


#plt.title("Convergencia del Gradiente (||∇f||)")
plt.xlabel("Iteraciones")
plt.ylabel("||∇f||")
plt.legend(fontsize="small", loc="upper right")
plt.show()


