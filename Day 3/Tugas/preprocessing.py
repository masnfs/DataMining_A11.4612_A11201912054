import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

print('cek data, Import Dataset')
print(X)
print(Y)

#from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print("\nimputer, Menghilangkan Missing Value (nan)")
print(X)

#from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print('\ncoltransformer, onehotencoder, Encoding data kategori (Atribut)')
print(X)

#from sklearn.preprocessing import LabelEncoder
laben = LabelEncoder()
Y = laben.fit_transform(Y)
print('\nlabel encoder, Encoding data kategori (Class / Label)')
print(Y)

#from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
print('\nre train_test_split, Membagi dataset ke dalam training set dan test set')
print('X_train\n', X_train)
print('X_test\n', X_test)
print('Y_train\n', Y_train)
print('Y_test\n', Y_test)

#from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print('\nfeature scaling')
print('X_train\n', X_train)
print('X_test\n', X_test)

print('\n\nDONE\n\n')