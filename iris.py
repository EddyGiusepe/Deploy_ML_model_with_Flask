import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('iris.data', header=None)
print("O shape de meus Dados são: ", df.shape)
print("Vejamos alguns de nossos Dados:")
print(df.head())

x =np.array(df.iloc[:, 0:4])
y =np.array(df.iloc[:, 4:])
print("Shape de meu atributos (features): ", x.shape)
print("Shape de minha Target: ", y.shape)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

print("shape de x_train:", x_train.shape)
print("shape de x_test:", x_test.shape)
print("shape de y_train:", y_train.shape)
print("shape de y_test:", y_test.shape)

from sklearn.svm import SVC
sv = SVC(kernel='linear').fit(x_train, y_train)

y_pred = sv.predict(x_test)
print(y_pred[:10])
print(y_test[:10])

from sklearn import metrics
print("Nossa accuracy é: {:0.2f}".format(metrics.accuracy_score(y_test, y_pred)))

# Salvamos nosso modelo treinado
pickle.dump(sv, open('iri.pkl', 'wb'))












