import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from io import StringIO
from IPython.display import Image, display
import pydotplus

sns.set()

test_df = pd.read_csv('titanic-test.csv')
train_df = pd.read_csv('titanic-train.csv')
# print(train_df.head(10))

train_df.Sex.value_counts().plot(kind='bar', color=['b', 'r'])
plt.title('Distribucion de sobrevivientes')
# plt.show()

label_encoder = preprocessing.LabelEncoder()

encoder_sex = label_encoder.fit_transform(train_df['Sex'])

print(train_df.head(10))

train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Embarked'] = train_df['Embarked'].fillna('S')

train_predictors = train_df.drop(
    ['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)

categorical_cols = [cname for cname in train_predictors.columns if train_predictors[cname].nunique(
) < 10 and train_predictors[cname].dtype == 'object']

numerical_cols = [cname for cname in train_predictors.columns if
                  train_predictors[cname].dtype in ['int64', 'float64']
                  ]

my_cols = categorical_cols+numerical_cols

train_predictors = train_predictors[my_cols]

dummy_encoded_train_predictors = pd.get_dummies(train_predictors)

print(train_predictors['Pclass'].value_counts())

y_target = train_df['Survived'].values
x_features_one = dummy_encoded_train_predictors.values

x_train, x_validation, y_train, y_validation = train_test_split(
    x_features_one, y_target, test_size=.25, random_state=1)

tree_one = tree.DecisionTreeClassifier()
tree_one = tree_one.fit(x_features_one, y_target)

tree_one_accurracy = round(tree_one.score(x_features_one, y_target), 4)

print('Accuracy: %0.4f' % (tree_one_accurracy))

out = StringIO()
tree.export_graphviz(tree_one, out_file=out)

graph = pydotplus.graph_from_dot_data(out.getvalue())
graph.write_png('titanic.png')
