# Importamos as nossas bibliotecas

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle

# Load the csv file
df = pd.read_csv('iris.csv')
print("Vejamos o Shape de nosso DataSet: ", df.shape)

# Print de nosso DataSet
print(df.head(10))

# Vamos mostrar os nomes de nossas colunas
print("As nossas colunas são: ", df.columns)

# Select independent and dependent variable
x = df[['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']]
y = df['Class']

print(x.shape)
print(y.shape)

# Split the DataSet into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50)
# random_state=50 --> para que nosso resultado seja o mesmo do que dele

# Feature scaling (to standardize our independent variable )
sc = StandardScaler() # Aqui instanciamos a classe StandardScaler
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# Scale data to have mean 0 and variance 1
# which is importance for convergence of the neural network
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
x_train_df = pd.DataFrame(x_train)
print(x_train_df)
print("Média igual (zero): ", np.mean(x_train_df[0]))
print("Variância igual (um): ", np.std(x_train_df[0]))

# Instantiate the model
classifier = RandomForestClassifier()

# Fit the model and we apply some predictions
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
#print(y_pred[:10], y_test[:10])
#print(y_test[:10])
out_RFC = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
print("A seguir comparamos nossos Dados de teste com as nossas predições: ")
print(out_RFC.head(10))

# Accuracy the our model
print("Nossa accuracy é: {:0.2f}".format(metrics.accuracy_score(y_test, y_pred)))

# Make pickle file of our model
pickle.dump(classifier, open('model.pkl', 'wb'))
