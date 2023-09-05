import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from RandomForest import RandomForest  
from math import sqrt

def ejecutar_random_forest():
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
    print(y_test)

    # Construir un Random Forest y entrenarlo
    num_trees = 100
    max_depth = 11
    max_features = int(sqrt(X.shape[1]))
    random_forest = RandomForest(num_trees=num_trees, max_depth=max_depth, max_features=max_features)
    random_forest.fit(X_train, y_train)

    # Hacer predicciones en el conjunto de prueba
    predictions = random_forest.predict(X_test)
    print(predictions)

    # Calcular la precisión
    accuracy = accuracy_score(y_test, predictions)
    print("Precisión del Random Forest:", accuracy)

    # Mostrar el informe de clasificación
    classification_rep = classification_report(y_test, predictions)
    print("Informe de clasificación del Random Forest:\n", classification_rep)

    # Crear la matriz de confusión
    cm = confusion_matrix(y_test, predictions)

    # Crear un DataFrame para la matriz de confusión
    df_cm = pd.DataFrame(cm, index=["0", "1"], columns=["0", "1"])

    # Visualizar la matriz de confusión
    plt.figure(figsize=(5, 5))
    sns.heatmap(df_cm, annot=True, cmap="Greens", fmt='.0f', cbar=False, annot_kws={"size": 14})
    plt.xlabel("Etiqueta Predicha")
    plt.ylabel("Etiqueta Real")
    plt.title("Matriz de Confusión del Random Forest")
    plt.show()

if __name__ == "__main__":
    num_repeticiones = 5  # Cambia el número de repeticiones según tus necesidades
    for i in range(num_repeticiones):
        print(f"Ejecución {i + 1}:")
        ejecutar_random_forest()
