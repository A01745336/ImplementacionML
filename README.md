# ImplementacionML
Momento de Retroalimentación: Módulo 2 Implementación de una técnica de aprendizaje máquina sin el uso de un framework. (Portafolio Implementación)

# Manual de Usuario: Árbol de Decisión Simple en Python
Este manual proporciona instrucciones sobre cómo usar el código para construir y evaluar un árbol de decisión simple en Python.

# Requisitos Previos
Python 3.x instalado en tu computadora.

Las bibliotecas numpy y scikit-learn instaladas. Puedes instalarlas utilizando el siguiente comando:

Copy code
pip install numpy scikit-learn

# Paso 1: Descargar el Código
Descarga el código proporcionado (el cual incluye las clases y funciones necesarias para el árbol de decisión) y guárdalo en un archivo con extensión .py.

# Paso 2: Preparar los Datos
En el archivo, busca la sección que contiene los datos de entrada. Puedes modificar la matriz X para incluir tus características y el vector y para incluir tus etiquetas. Asegúrate de que la cantidad de elementos en X coincida con la cantidad de elementos en y.

# Paso 3: Construir el Árbol de Decisión
Ejecuta el archivo en tu entorno de Python.
El código dividirá automáticamente los datos en conjuntos de entrenamiento y prueba utilizando la función train_test_split.
Luego, construirá el árbol de decisión utilizando los datos de entrenamiento y la función build_tree.

# Paso 4: Evaluar el Árbol de Decisión
Después de construir el árbol, el código lo evaluará utilizando el conjunto de prueba.
Calculará la exactitud (accuracy) del árbol en el conjunto de prueba y la imprimirá en la pantalla.

# Paso 5: Realizar Predicciones
Si deseas realizar predicciones personalizadas con el árbol construido:

Crea una lista new_data_point que contenga las características para las cuales deseas hacer una predicción.
Llama a la función predict(tree, new_data_point) pasando el árbol construido y la lista new_data_point.

# Ejemplo proporcionado en el código
1. El nodo raíz (Depth: 0) divide los datos en función de la característica 0 (índice 0) y utiliza un umbral de 8.25. Esta división crea dos ramas: valores mayores que 8.25 y valores menores o iguales a 8.25.

2. En la rama izquierda (Depth: 1), donde los valores de la característica 0 son menores o iguales a 8.25, el nodo en Depth: 1 utiliza la característica 1 (índice 1) y un umbral de 4.75 para dividir los datos en dos ramas: valores mayores que 4.75 y valores menores o iguales a 4.75.

3. En la primera rama de la rama izquierda (Depth: 2), donde los valores de la característica 1 son menores o iguales a 4.75, el nodo en Depth: 2 utiliza nuevamente la característica 0 y un umbral de 7.0 para dividir los datos en dos ramas. En este nivel, las hojas indican que los valores pertenecen a la clase 0 o 1 dependiendo de si hay más valores de una clase que de otra.

4. En la segunda rama de la rama izquierda (Depth: 2), donde los valores de la característica 1 son mayores que 4.75, el nodo en Depth: 2 utiliza la característica 0 y un umbral de 6.5 para dividir los datos en dos ramas. Al igual que antes, las hojas indican las clases a las que pertenecen los valores.

5. En la rama derecha (Depth: 1), donde los valores de la característica 0 son mayores que 8.25, el nodo en Depth: 1 tiene hojas que indican directamente las clases a las que pertenecen los valores.

En resumen, el árbol de decisión toma decisiones basadas en características y umbrales para separar los datos en diferentes clases. Cuando haces una predicción para una nueva entrada, el árbol sigue estas decisiones y te proporciona una predicción basada en la ruta que sigue a través de las ramas del árbol.  Es importante señalar que debido a la construcción del árbol usar una mayor o menor cantidad de "Depth" puede alterar el resultado de la predicción esperada debido a que es posible que la profundidad adicional esté permitiendo que el árbol capture más detalles ruidosos de los datos y, por lo tanto, esté resultando en predicciones menos precisas en nuevos datos.  Los casos de ejemplo estan diseñados para que den las predicciones esperadas usando 3 depths.

# Se proporciona otro archivo con otro árbol de ejemplo

Para usar el ejemplo 2, es necesario tener este archivo junto con el archivo "AbolDecision.py" en la misma carpeta y correr unicamente el archivo "ejemplo2".  En este nuevo archivo, se proporciona un arbol diferente y una ofrma en la que el usuario pueda hacer las predicciones que quiera sin necesidad de modificar el codigo base.

## Manual de Usuario: Implementación de Random Forest en Python

Este manual proporciona instrucciones sobre cómo usar la implementación de Random Forest en Python para construir y evaluar un modelo de Random Forest en el conjunto de datos del Titanic.

# Requisitos Previos
- Python 3.x instalado en tu computadora.
- Las bibliotecas numpy, scikit-learn y seaborn instaladas. Puedes instalarlas utilizando el siguiente comando:

```bash
pip install numpy scikit-learn seaborn
```

### Paso 1: Descargar el Código
Descarga el código proporcionado (`RandomForest.py` y `ArbolDecision.py`) y guárdalo en un archivo con extensión `.py`.

### Paso 2: Preparar los Datos
En el archivo, se realiza el preprocesamiento de los datos del Titanic. Asegúrate de que el código esté cargando el conjunto de datos y realizando las transformaciones adecuadas.

### Paso 3: Construir y Entrenar el Random Forest
Ejecuta el archivo en tu entorno de Python. El código dividirá automáticamente los datos en conjuntos de entrenamiento y prueba utilizando la función `train_test_split`. Luego, construirá y entrenará el Random Forest utilizando los datos de entrenamiento y la clase `RandomForest`.

### Paso 4: Hacer Predicciones y Evaluar el Modelo
Después de entrenar el Random Forest, el código hará predicciones en el conjunto de prueba y calculará la precisión del modelo utilizando la función `accuracy_score`. La precisión del modelo en el conjunto de prueba se imprimirá en la pantalla.

### Paso 5: Experimentar con Hiperparámetros (Opcional)
Dentro del archivo `RandomForest.py`, encontrarás la clase `RandomForest` que contiene varios hiperparámetros. Puedes ajustar estos hiperparámetros para experimentar con diferentes configuraciones del modelo. Algunos hiperparámetros clave incluyen:
- `num_trees`: Número de árboles en el bosque.
- `max_depth`: Profundidad máxima de cada árbol.
- `max_features`: Cantidad máxima de características consideradas en cada división.
- `random_state`: Valor para garantizar reproducibilidad.

### Paso 6: Interpretar los Resultados
Observa la precisión del modelo en el conjunto de prueba y compárala con diferentes configuraciones de hiperparámetros. Si la precisión parece ser baja, considera ajustar los hiperparámetros o explorar técnicas de validación cruzada para una evaluación más exhaustiva.

### Nota
Debido a la aleatoridad del modelo, puede llegar a suceder que en algunas corridas el modelo "empeore" su precisión, mientras que en otras mejore de manera drastica

### Ejemplo Proporcionado
El archivo de ejemplo proporcionado muestra cómo construir y entrenar un Random Forest en el conjunto de datos del Titanic. También realiza predicciones y calcula la precisión del modelo. Asegúrate de seguir los pasos anteriores y ajustar los hiperparámetros según sea necesario para mejorar el rendimiento del modelo en este conjunto de datos específico.