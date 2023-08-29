import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from RandomForest import RandomForest

# Cargar el conjunto de datos Titanic
titanic = sns.load_dataset("titanic")

# Preprocesamiento básico del conjunto de datos
titanic.dropna(inplace=True)  # Eliminar filas con datos faltantes
titanic = titanic[["pclass", "sex", "age", "sibsp", "parch", "fare", "survived"]]
titanic["sex"] = titanic["sex"].map({"male": 0, "female": 1})

# Dividir el conjunto de datos en características (X) y etiquetas (y)
X = titanic.drop("survived", axis=1).values
y = titanic["survived"].values

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construir un Random Forest y entrenarlo
num_trees = 100
max_depth = 25
max_features = 6
random_forest = RandomForest(num_trees, max_depth, max_features)
random_forest.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
predictions = []
for i in range(len(X_test)):
    prediction = random_forest.predict(X_test[i])
    predictions.append(prediction)
