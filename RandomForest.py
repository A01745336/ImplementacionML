from ArbolDecision import TreeNode, build_tree, predict
import numpy as np
from sklearn.metrics import accuracy_score
from collections import Counter


class RandomForest:
    def __init__(self, num_trees, max_depth=None, max_features=None):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        train_accuracy_prev = 0  # Almacenar la precisión previa
        for i in range(self.num_trees):
            # Seleccionar un subconjunto aleatorio de características
            random_feature_indices = np.random.choice(len(X[0]), size=self.max_features, replace=False)
            X_train_subset = X[:, random_feature_indices]

            # Seleccionar un subconjunto aleatorio de datos
            random_indices = np.random.choice(len(X_train_subset), size=len(X_train_subset), replace=True)
            X_train_subset = X_train_subset[random_indices]
            y_train_subset = y[random_indices]

            # Construir un árbol de decisión usando tu implementación
            tree = build_tree(X_train_subset, y_train_subset, max_depth=self.max_depth, max_features=self.max_features)

            self.trees.append(tree)

            # Evaluar la precisión en el conjunto de entrenamiento
            train_predictions = [predict(tree, x) for x in X_train_subset]
            train_accuracy = accuracy_score(y_train_subset, train_predictions)

            # Calcular la mejora en la precisión
            improvement = train_accuracy - train_accuracy_prev

            # Mostrar la información (opcional)
            # print(f'Árbol {i+1} entrenado. Precisión en el conjunto de entrenamiento: {train_accuracy:.4f}')
            # print(f'Mejora del modelo: {improvement:.4f}')

            # Actualizar la precisión previa
            # train_accuracy_prev = train_accuracy

    def predict(self, X):
        predictions = [self.predict_one(tree, x) for x, tree in zip(X, self.trees)]
        return predictions

    def predict_one(self, tree, x):
        if tree is None:
            # print("Árbol es None")
            pass
        if tree.left is None and tree.right is None:
            # print("Es una hoja, valor:", tree.value)
            return tree.value
        # print("Comparando:", x[tree.feature_index], "con umbral:", tree.threshold)
        if x[tree.feature_index] <= tree.threshold:
            # print("Moviendo hacia la izquierda")
            return self.predict_one(tree.left, x)
        else:
            # print("Moviendo hacia la derecha")
            return self.predict_one(tree.right, x)



    def evaluate(self, X, y):
        predictions = [self.predict(x) for x in X]
        accuracy = accuracy_score(y, predictions)
        print(f'Precisión del Random Forest: {accuracy:.4f}')
