import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm 

# 1. Cargar el dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Añadir una columna de 1s a X para la ordenada al origen
X = np.hstack((np.ones((X.shape[0], 1)), X))

# 2. Dividir el dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Estandarizar los datos (excepto la columna de 1s)
scaler = StandardScaler()
X_train[:, 1:] = scaler.fit_transform(X_train[:, 1:])
X_test[:, 1:] = scaler.transform(X_test[:, 1:])

# Implementar cuadrados minimos
# 4. Pseudoinversa
def pseudoinverse_solution(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y

# Implementar gradiente descendiente, emplear tasa igual a 1/sigma1^2, sigma1 siendo primer valor singular de X, pq tiene sentido usar ese valor?
# 5. Descenso por gradiente
def gradient_descent(X, y, learning_rate, iterations):
    w = np.zeros(X.shape[1])  # Inicialización
    n = X.shape[0]
    errors = []
    older_gradient = None
    for i in tqdm(range(iterations)):
        gradient = -(2 / n) * X.T @ (y - X @ w)
        w = w - learning_rate * gradient
        error = np.mean((y - X @ w) ** 2)  # Error cuadrático medio
        errors.append(error)
        if (older_gradient is not None and np.linalg.norm(gradient - older_gradient) < 1e-6):
            break
        older_gradient = gradient

    return w, errors


# Calcular η utilizando el valor singular más grande de X_train
sigma1 = np.linalg.svd(X_train, compute_uv=False)[0]
learning_rate = 1 / (sigma1 ** 2)

# Ejecutar el descenso por gradiente
iterations = 10000000
w_gradient, errors = gradient_descent(X_train, y_train, learning_rate, iterations)


# 6. Comparar soluciones
def evaluate_model(X, y, w):
    y_pred = X @ w
    mse = np.mean((y - y_pred) ** 2)
    return mse

w_pseudo = pseudoinverse_solution(X_train, y_train)
train_error_pseudo = evaluate_model(X_train, y_train, w_pseudo)
test_error_pseudo = evaluate_model(X_test, y_test, w_pseudo)

train_error_gradient = evaluate_model(X_train, y_train, w_gradient)
test_error_gradient = evaluate_model(X_test, y_test, w_gradient)

# Mostrar errores
print("Pseudoinversa:")
print(f"Error en entrenamiento: {train_error_pseudo:.4f}")
print(f"Error en testeo: {test_error_pseudo:.4f}")

print("\nDescenso por gradiente:")
print(f"Error en entrenamiento: {train_error_gradient:.4f}")
print(f"Error en testeo: {test_error_gradient:.4f}")

# 7. Gráfica del error del descenso por gradiente
plt.plot(range(len(errors)), errors)
plt.xlabel("Iteraciones")
plt.ylabel("Error cuadrático medio")
plt.title("Descenso por gradiente: Convergencia del error")
plt.show()

# Comparar solucion obtenida por la pseudoinversa con grad desc para distintos eta
# Mostrar error en conjunto de entrenamiento y de prueba frente a numero de iteraciones para grad desc