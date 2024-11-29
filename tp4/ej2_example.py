import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


# 4. Pseudoinversa
def pseudoinverse_solution(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y


w_pseudo = pseudoinverse_solution(X_train, y_train)


# 5. Descenso por gradiente
def gradient_descent(X, y, learning_rate, iterations):
    w = np.zeros(X.shape[1])  # Inicialización
    n = X.shape[0]
    errors = []

    for i in range(iterations):
        gradient = -(2 / n) * X.T @ (y - X @ w)
        w = w - learning_rate * gradient
        error = np.mean((y - X @ w) ** 2)  # Error cuadrático medio
        errors.append(error)

    return w, errors


# Calcular η utilizando el valor singular más grande de X_train
sigma1 = np.linalg.svd(X_train, compute_uv=False)[0]
learning_rate = 1 / (sigma1 ** 2)

# Ejecutar el descenso por gradiente
iterations = 500
w_gradient, errors = gradient_descent(X_train, y_train, learning_rate, iterations)


# 6. Comparar soluciones
def evaluate_model(X, y, w):
    y_pred = X @ w
    mse = np.mean((y - y_pred) ** 2)
    return mse


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
plt.plot(range(iterations), errors)
plt.xlabel("Iteraciones")
plt.ylabel("Error cuadrático medio")
plt.title("Descenso por gradiente: Convergencia del error")
plt.show()


# ---------------------------------------------------------------------
# REGULARIZACIÓN
def ridge_solution(X, y, lambda_):
    """Cálculo de la solución cerrada con Ridge Regression."""
    n_features = X.shape[1]
    I = np.eye(n_features)
    return np.linalg.inv(X.T @ X + lambda_ * I) @ X.T @ y


def ridge_gradient_descent(X, y, learning_rate, lambda_, iterations):
    """Descenso por gradiente con regularización L2."""
    w = np.zeros(X.shape[1])  # Inicialización
    n = X.shape[0]
    errors = []

    for i in range(iterations):
        gradient = -(2 / n) * X.T @ (y - X @ w) + 2 * lambda_ * w
        w = w - learning_rate * gradient
        error = np.mean((y - X @ w) ** 2)  # Error cuadrático medio
        errors.append(error)

    return w, errors


# Calcular el valor de lambda
sigma1 = np.linalg.svd(X_train, compute_uv=False)[0]
lambda_ = 10 ** -2 * sigma1

# Ridge Regression (solución cerrada)
w_ridge = ridge_solution(X_train, y_train, lambda_)

# Ridge Regression (descenso por gradiente)
w_ridge_gd, errors_ridge = ridge_gradient_descent(X_train, y_train, learning_rate, lambda_, iterations)

# Comparar errores
train_error_ridge = evaluate_model(X_train, y_train, w_ridge)
test_error_ridge = evaluate_model(X_test, y_test, w_ridge)

train_error_ridge_gd = evaluate_model(X_train, y_train, w_ridge_gd)
test_error_ridge_gd = evaluate_model(X_test, y_test, w_ridge_gd)

# Mostrar resultados
print("Ridge Regression (Solución Cerrada):")
print(f"Error en entrenamiento: {train_error_ridge:.4f}")
print(f"Error en testeo: {test_error_ridge:.4f}")

print("\nRidge Regression (Descenso por Gradiente):")
print(f"Error en entrenamiento: {train_error_ridge_gd:.4f}")
print(f"Error en testeo: {test_error_ridge_gd:.4f}")

# Gráfica del error para Ridge Regression con gradiente
plt.plot(range(iterations), errors_ridge)
plt.xlabel("Iteraciones")
plt.ylabel("Error cuadrático medio")
plt.title("Ridge Regression: Convergencia del error")
plt.show()
