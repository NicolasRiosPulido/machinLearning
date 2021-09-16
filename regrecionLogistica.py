import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pylab as plt
import seaborn as sns
# %matplotlib inline

diabetes = pd.read_csv('diabetes.csv')

feature_cols = ['Pregnancies', 'Insulin',
                'BMI', 'Age', 'Glucose', 'BloodPressure']

x = diabetes[feature_cols]
y = diabetes.Outcome

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(y_pred)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

class_names = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

print('Exactitud ', metrics.accuracy_score(y_test, y_pred)*100, '%')

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='Blues_r', fmt='g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Matriz de confucion', y=1.1)
plt.ylabel('Etiqueta actual')
plt.xlabel('Etiqueta de prediccion')
plt.show()
