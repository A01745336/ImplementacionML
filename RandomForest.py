from ArbolDecision import TreeNode, build_tree, predict
import numpy as np
from sklearn.metrics import accuracy_score


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

            tree = build_tree(X_train_subset, y_train_subset, max_depth=self.max_depth, max_features=self.max_features)
            self.trees.append(tree)

            # Evaluar la precisión en el conjunto de entrenamiento
            train_predictions = [self.predict(x, tree_idx=i) for x in X_train_subset]
            train_accuracy = accuracy_score(y_train_subset, train_predictions)

            # Calcular la mejora en la precisión
            improvement = train_accuracy - train_accuracy_prev

            # Mostrar la información
            print(f'Árbol {i+1} entrenado. Precisión en el conjunto de entrenamiento: {train_accuracy:.4f}')
            print(f'Mejora del modelo: {improvement:.4f}')

            # Actualizar la precisión previa
            train_accuracy_prev = train_accuracy

    def predict(self, x, tree_idx=None):
        predictions = []
        for idx, tree in enumerate(self.trees):
            if tree_idx is None or tree_idx == idx:
                prediction = predict(tree, x)
                predictions.append(prediction)
        # Aplicar votación o promediado para combinar las predicciones
        final_prediction = np.round(np.mean(predictions))
        return final_prediction

    def evaluate(self, X, y):
        predictions = [self.predict(x) for x in X]
        accuracy = accuracy_score(y, predictions)
        print(f'Precisión del Random Forest: {accuracy:.4f}')
