from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

# Data frame
perfiles = pd.read_csv('data_otto_test.csv', sep=';')

total_deuda = perfiles['Ahorro']-perfiles['Deuda']
total_ingresos = perfiles['Ingrasos']-perfiles['Gastos']
#perfiles = perfiles.assign(total_deuda=total_deuda,total_ingresos=total_ingresos)

# K-means
model = KMeans(n_clusters=2, max_iter=1000)
model.fit(perfiles)
y_labels = model.labels_


y_kmeans = model.predict(perfiles)
print('Predicciones ', y_kmeans)

plt.scatter(perfiles['Deuda'], perfiles['Ahorro'], c=y_kmeans, s=30)
plt.xlabel('Deuda', fontsize=10)
plt.ylabel('Ahorro', fontsize=10)
plt.show()
