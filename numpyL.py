import numpy as np

arr = np.array([1, 2, 3, 4, 5])

print(arr[2:])

print(arr[0::2])

# crea arreglo de zeros
# zeros vector
c = np.zeros(5)
print(c)

# unos matriz
m = np.ones((4, 5))
print(m)

# parametros inicio, fin, numero de datos entre fin he inicio
l = np.linspace(3, 10, 5)
print(l)

# sort con arreglo complejo

cabecera = [('nombre', 'S10'), ('edad', int)]
datos = [('juan', 10), ('Maria', 70), ('Javier', 42), ('Samuel', 15)]
usuarios = np.array(datos, dtype=cabecera)

order = np.sort(usuarios, order='edad')

print(order)

# crear arreglo

ac = np.arange(25)
print(ac)
arregloIntervalos = np.arange(5, 50, 5)
print(arregloIntervalos)

# matris

m = np.full((3, 5), 10)
print(m)

# matris identidad

mi = np.diag([1, 1, 1, 1, 1])
print(mi)
