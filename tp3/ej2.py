import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cargar datos
X = pd.read_csv('C:/Users/marco/Desktop/Udesa/Metodos/Tp3/dataset01.csv')
X = X.iloc[:, 1:].values  # Eliminar la primera columna
y = pd.read_csv('C:/Users/marco/Desktop/Udesa/Metodos/Tp3/y1.txt', header=None).squeeze().values  # Cargar y.txt y convertirlo en un array unidimensional

print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Probar diferentes dimensiones d
dimensions = [2, 6, 10, 50, 100, 150, X.shape[1]]
pca_results = {}

for d in dimensions:
    pca = PCA(n_components=d)
    Z = pca.fit_transform(X)
    pca_results[d] = (Z, pca)

    print(f"PCA with d={d}: explained variance ratio: {pca.explained_variance_ratio_}")

    # Gráfico 2D
    Z_2d = pca_results[d][0]

    plt.figure(figsize=(10, 8))
    plt.scatter(Z_2d[:, 0], Z_2d[:, 1], cmap='winter', s=10)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.show()

    if d > 2:
        # Gráfico 3D
        Z_3d = pca_results[d][0]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(Z_3d[:, 0], Z_3d[:, 1], Z_3d[:, 2], cmap='winter', s=10)
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')
        plt.show()

# Visualización de varianza explicada acumulativa
explained_variances = [pca_results[d][1].explained_variance_ratio_.sum() for d in dimensions]
plt.plot(dimensions, explained_variances, marker='o')
plt.xlabel('Numero de dimensiones d')
plt.ylabel('Varianza explicada acumulativa')
plt.show()

# Función para calcular la matriz de similaridad
def similarity_matrix(X, sigma=1.0):
    dist_matrix = euclidean_distances(X)
    sim_matrix = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))
    return sim_matrix

# Matrices de similaridad
similarity_matrices = {d: similarity_matrix(pca_results[d][0]) for d in dimensions}
original_similarity_matrix = similarity_matrix(X)

# Visualización de matrices de similaridad
plt.figure(figsize=(10, 8))
sns.heatmap(original_similarity_matrix, cmap='viridis')
plt.show()

for d in dimensions:
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrices[d], cmap='viridis')
    plt.show()

# Importancia de las dimensiones originales
for d in dimensions:
    Z, pca = pca_results[d]
    most_important_features = np.abs(pca.components_).sum(axis=0)
    sorted_features = np.argsort(most_important_features)[::-1]

    print(f"Most important features for d={d}:")
    for idx in sorted_features[:5]:  # Mostramos las 5 más importantes
        print(f"Feature {idx}: {most_important_features[idx]}")

# Función para evaluar regresión lineal y calcular el error cuadrático medio
def evaluate_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    return mse, model.coef_, y_pred  

# Evaluamos el error para diferentes dimensiones d
mse_results = {}

for d in dimensions:
    Z, _ = pca_results[d]
    mse, coefs, y_pred = evaluate_regression(Z, y)
    mse_results[d] = mse
    print(f"MSE for d={d}: {mse}")
    print(f"Regression coefficients for d={d}: {coefs}")

# Visualización de errores
plt.plot(dimensions, [mse_results[d] for d in dimensions], marker='o')
plt.xlabel('Numero de dimensiones d')
plt.ylabel('Error cuadrático medio')
plt.show()

# Realizar predicción con d=2
Z_2d, _ = pca_results[2]
_, _, y_pred_2d = evaluate_regression(Z_2d, y)

# Calcular errores individuales
errors_2d = np.abs(y - y_pred_2d)

# Ordenar muestras por menor error y mostrar las de mejor predicción
best_predictions_indices = np.argsort(errors_2d)[:10]  
print("Índices de las muestras con mejor predicción para d=2:")
print(best_predictions_indices)
print("Errores de las mejores predicciones para d=2:")
print(errors_2d[best_predictions_indices])
