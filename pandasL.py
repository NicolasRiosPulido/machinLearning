import pandas as pd

series = pd.Series([5, 10, 15, 20, 25])
print(series)

lst = ['hola', 'mundo', 'robotico']
df = pd.DataFrame(lst)
print(df)

data = {'Nombre': ['Juan', 'Ana', 'Jose', 'Arturo'],
        'Edad': [25, 18, 23, 27],
        'pais': ['MX', 'CO', 'BR', 'MX']}

dfB = pd.DataFrame(data)
print(dfB)
print(dfB[['Nombre']])

dataf = pd.read_csv('canciones.csv')
print(dataf.head(5))

#artista = dataf.artists
# print(artista)
# print(dataf.shape)
