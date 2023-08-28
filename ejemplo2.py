from ArbolDecision import build_tree, predict
import numpy as np

if __name__ == "__main__":
    # Ejemplo de datos de entrada (características y etiquetas)
    X = np.array([
        [6.5],
        [4.5],
        [6.0],
        [8.0],
        [9.0],
        [4.0],
        [7.0],
        [5.0],
        [3.0],
        [2.0],
        [7.5],
        [6.0],
        [8.5],
        [1.5]
    ])
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0])

    # Construir el árbol de decisión
    tree = build_tree(X, y, max_depth=3)

    # Ejemplo de nuevos datos de entrada (características)
    num_predictions = int(input("Ingrese la cantidad de predicciones que desea hacer: "))
    X_new = []
    for _ in range(num_predictions):
        feature_value = float(input("Ingrese el valor de la característica: "))
        X_new.append([feature_value])

    X_new = np.array(X_new)

    # Hacer predicciones utilizando el nuevo árbol construido
    for i in range(len(X_new)):
        prediction_new = predict(tree, X_new[i])
        print("Predicción para la entrada", i+1, ":", prediction_new)
