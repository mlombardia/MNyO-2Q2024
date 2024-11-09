import numpy as np
import os
from PIL import Image
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
# Aprender representacion basada en SVD utilizando las n imagenes
# Visualizar en pxp las imagenes reconstruidas luego de compresión con distintos valores de d; Conclusiones?
# Analizar evolucion de error de cada imagen comprimida y su original sin exceder el 10% bajo la norma de Frobenius;
# ¿Qué dimensión asegura el error menor al 10%?

def load_images_from_folder(folder_path, image_extension='jpeg'):
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(image_extension)])
    images = []
    for file_name in file_list:
        img_path = os.path.join(folder_path, file_name)
        img = Image.open(img_path).convert('L')  # Convertir a escala de grises
        img_array = np.array(img).flatten()
        images.append(img_array)
    images = np.array(images)
    return images

# Cargar imágenes de dataset_imgs
images1 = load_images_from_folder('./dataset_images')
n1, p2 = images1.shape
p = int(np.sqrt(p2))
print(f"Loaded {n1} images of size {p}x{p}")

# Probar diferentes dimensiones d
dimensions = [2, 6, 10, 50, 100, 500, p2]
svd_results1 = {}
for d in dimensions:
    svd = TruncatedSVD(n_components=d)
    Z = svd.fit_transform(images1)
    svd_results1[d] = (Z, svd)
    print(f"SVD with d={d}: explained variance ratio: {svd.explained_variance_ratio_.sum()}")

# Función para reconstruir imágenes desde su representación de baja dimensión
def reconstruct_images(Z, svd, original_shape):
    X_approx = svd.inverse_transform(Z)
    images_approx = X_approx.reshape((-1, *original_shape))
    return images_approx

# Visualizar imágenes reconstruidas para diferentes dimensiones d
def plot_reconstructed_images(images, title, num_images=5):
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

# Mostrar imágenes reconstruidas
for d in dimensions:
    Z, svd = svd_results1[d]
    reconstructed_images = reconstruct_images(Z, svd, (p, p))
    plot_reconstructed_images(reconstructed_images, f'Reconstructed Images with d={d}')
# Función para calcular la matriz de similaridad
def similarity_matrix(X, sigma=800.0):
    dist_matrix = euclidean_distances(X)
    sim_matrix = np.exp(-dist_matrix**2 / (2 * sigma**2))
    return sim_matrix
# Matrices de similaridad
similarity_matrices = {d: similarity_matrix(svd_results1[d][0]) for d in dimensions}
original_similarity_matrix = similarity_matrix(images1)
# Visualización de matrices de similaridad
def plot_similarity_matrix(sim_matrix, title):
    plt.figure(figsize=(10, 8))
    plt.imshow(sim_matrix, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.show()
plot_similarity_matrix(original_similarity_matrix, 'Original Similarity Matrix')
for d in dimensions:
    plot_similarity_matrix(similarity_matrices[d], f'Similarity Matrix with d={d}')
# Función para calcular el error de reconstrucción con la norma de Frobenius
def frobenius_norm_error(X, X_approx):
    return np.linalg.norm(X - X_approx, 'fro') / np.linalg.norm(X, 'fro')
# Función para realizar SVD y reducir dimensionalidad
def svd_reduction(X, d):
    svd = TruncatedSVD(n_components=d)
    Z = svd.fit_transform(X)
    return Z, svd
# Encontrar el número mínimo de dimensiones d para imagenes
errors = []
optimal_d = 0
for d in range(1, 100):
    Z, svd = svd_reduction(images1, d)
    reconstructed_images = svd.inverse_transform(Z)
    error = frobenius_norm_error(images1, reconstructed_images)
    if error <= 0.00001:
        break
    print(f"Frobenius norm error: {error*100:.2f}% with d={d}")
    errors.append(error)
    if error <= 0.10 and optimal_d == 0:
        optimal_d = d

print(f"Minimum number of dimensions to keep the error <= 10%: {optimal_d}")