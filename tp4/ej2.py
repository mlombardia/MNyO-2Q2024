import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm 

data = fetch_california_housing()
X, y = data.data, data.target

X = np.hstack((np.ones((X.shape[0], 1)), X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train[:, 1:] = scaler.fit_transform(X_train[:, 1:])
X_test[:, 1:] = scaler.transform(X_test[:, 1:])

# -------------------Implementar cuadrados minimos --------------------------
def pseudoinverse_solution(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y

# --------------Implementar gradiente descendiente, emplear tasa igual a 1/sigma1^2, sigma1 siendo primer valor singular de X, pq tiene sentido usar ese valor?---------------------------
def gradient_descent(X, y, learning_rate, iterations):
    w = np.zeros(X.shape[1])  # Inicialización
    n = X.shape[0]
    errors = []
    older_gradient = None
    for i in tqdm(range(iterations)):
        gradient = -(2 / n) * X.T @ (y - X @ w)
        w = w - learning_rate * gradient
        error = np.mean((y - X @ w) ** 2)  
        errors.append(error)
        if (older_gradient is not None and np.linalg.norm(gradient - older_gradient) < 1e-10):
            break
        older_gradient = gradient

    return w, errors


sigma1 = np.linalg.svd(X_train, compute_uv=False)[0]
sigman = np.linalg.svd(X_train, compute_uv=False)[-1]
learning_rate = 1 / ((sigma1) ** 2)
print(f"learning_rate: {learning_rate}")

iterations = [10000,25000,50000,100000,250000,500000]
w_gradient_list = []
w_gradient_error_list = []
for it in iterations:
    w_gradient, errors = gradient_descent(X_train, y_train, learning_rate, it)
    w_gradient_list.append(w_gradient)

# --------------Comparar solucion obtenida por la pseudoinversa con grad desc para distintos eta----------------------
def evaluate_model(X, y, w):
    y_pred = X @ w
    mse = np.mean((y - y_pred) ** 2)
    return y_pred, mse

w_pseudo_list = []
w_pseudo_error_list = []
for it in iterations:
    w_pseudo = pseudoinverse_solution(X_train, y_train)
    w_pseudo_list.append(w_pseudo)

for pseudo in w_pseudo_list:
    _,train_error_pseudo = evaluate_model(X_train, y_train, pseudo)
    pseudo_pred, test_error_pseudo = evaluate_model(X_test, y_test, pseudo)
    w_pseudo_error_list.append(test_error_pseudo)

for gradient in w_gradient_list:
    _,train_error_gradient = evaluate_model(X_train, y_train, gradient)
    gradient_pred,test_error_gradient = evaluate_model(X_test, y_test, gradient)
    w_gradient_error_list.append(test_error_gradient)

# ---------------Mostrar error en conjunto de entrenamiento y de prueba frente a numero de iteraciones para grad desc-----------------------
print("Pseudoinversa:")
print(f"Error en entrenamiento: {train_error_pseudo:.4f}")
print(f"Error en testeo: {test_error_pseudo:.4f}")

print("\Gradiente descendiente:")
print(f"Error en entrenamiento: {train_error_gradient:.4f}")
print(f"Error en testeo: {test_error_gradient:.4f}")

indices = range(len(iterations))
plt.plot(indices, w_gradient_error_list)
plt.xticks(indices,iterations)
plt.xlabel("Iteraciones")
plt.ylabel("Error cuadrático medio")
plt.show()

plt.plot(range(iterations[0]), errors[0])
plt.xlabel("Iteraciones")
plt.ylabel("Error cuadrático medio")
plt.show()