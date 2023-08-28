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
