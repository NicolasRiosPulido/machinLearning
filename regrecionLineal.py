import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sb

dataset = pd.read_csv('salarios.csv')
#perfiles = pd.read_csv('data_otto.csv', usecols=['Deuda', 'Ahorro'])

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

print(x.shape)
print(y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

# print(X_train)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)
viz_train = plt
viz_train.scatter(X_test, Y_test, color='blue')
viz_train.plot(X_train, regressor.predict(X_train), color='green')
viz_train.title('Salario Vs Experiencia')
viz_train.xlabel('Experiencia')
viz_train.ylabel('Salario')
viz_train.show()
