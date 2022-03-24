# Importamos as nossas bibliotecas

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the csv file
df = pd.read_csv('iris.csv')
print("Vejamos o Shape de nosso DataSet: ", df.shape)
print(df.head())
print("As nossas colunas são: ", df.columns)

#