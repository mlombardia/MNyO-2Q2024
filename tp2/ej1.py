import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargo el archivo CSV
data = pd.read_csv('city_temperature.csv')

# Filtro data sucia y sumo las columnas util
filtered_data = data[(data['Year'] >= 1995) & (data['Day'] > 0) & (data['AvgTemperature'] > -99)]

filtered_data['Date'] = pd.to_datetime(filtered_data[['Year', 'Month', 'Day']])
filtered_data = filtered_data.sort_values(by=['City', 'Date'])

# Calculo los deltas
filtered_data['delta_temp'] = filtered_data['AvgTemperature'].diff()
filtered_data['delta_time'] = filtered_data['Date'].diff().dt.total_seconds() / (60 * 60 * 24)  # Convertir a días

# Calcular la tasa de cambio de temperatura
filtered_data['rate_of_change'] = filtered_data['delta_temp'] / filtered_data['delta_time']


def similitude(series_i, series_j):
    # Calcular la métrica de similitud (distancia euclidiana entre las tasas de variación)
    similitude = math.sqrt(np.sum(np.abs(series_i - series_j) ** 2))
    return similitude


region1 = filtered_data[(filtered_data['City'] == 'La Paz') & (filtered_data['Year'] == 2010)]
region2 = filtered_data[(filtered_data['City'] == 'Singapore') & (filtered_data['Year'] == 2010)]

# Equiparo las lineas de tiempo
min_length = min(len(region1), len(region2))
region1 = region1[:min_length]
region2 = region2[:min_length]

similarity_score = similitude(region1['rate_of_change'].values, region2['rate_of_change'].values)

print(similarity_score)

# Gráfico de series temporales de rate_of_change y temperatura promedio para paises en regiones
regions = [region1, region2]
for region in regions:
    plt.plot(region['Date'], region['rate_of_change'], label=region['City'].iloc[0])

plt.xlabel('Fecha', fontsize=14)
plt.ylabel('Tasa de cambio (°F)', fontsize=14)
plt.legend()
plt.show()

for region in regions:
    plt.plot(region['Date'], region['AvgTemperature'], label=region['City'].iloc[0])

plt.xlabel('Fecha', fontsize=14)
plt.ylabel('Temperatura promedio (°F)', fontsize=14)
plt.legend()
plt.show()

