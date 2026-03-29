#  Clasificador de Datos con Machine Learning

Este script implementa un sistema de clasificación de datos en Python utilizando diferentes algoritmos de Machine Learning como:

- k-Nearest Neighbors (kNN)
- Árboles de Decisión
- Random Forest

Incluye un pipeline completo de:
- Preprocesamiento de datos
- Entrenamiento
- Evaluación
- Predicción

Utiliza la versión de python 3.13.9

# Uso
El script se ejecuta desde línea de comandos y permite configurar distintos parámetros para el entrenamiento o test de modelos de clasificación.
- Si se ejecuta en modo train, el programa generará una carpeta output con un resumen .csv (modelo.csv) y el modelo guardado (modelo.pkl).
- Si se ejecuta en modo test, el programa usará automáticamente el modelo guardado en la carpeta output y generará en esa misma carpeta un .csv con los resultados (data-prediction.csv).

Argumentos disponibles:
- -m, --mode (obligatorio)
Modo de ejecución: train o test.
- -f, --file (obligatorio)
Ruta al fichero CSV con los datos.
- -a, --algorithm (obligatorio)
Algoritmo a utilizar: kNN, decision_tree o random_forest.
- -p, --prediction (obligatorio)
Nombre de la columna que se desea predecir.
- -e, --estimator (opcional)
Métrica para evaluar el modelo (según scikit-learn).
Por defecto: None.
- -c, --cpu (opcional)
Número de CPUs a utilizar (-1 para usar todas).
Por defecto: -1.
- -v, --verbose (opcional)
Muestra métricas por terminal.
Activar con flag (--verbose).
- --debug (opcional)
Activa el modo debug: muestra información adicional del preprocesado y guarda los resultados en un .csv.

Ejemplo de uso:
python clasificador.py -m train -f data.csv -a random_forest -p target --verbose


# Requerimientos

- Crear entorno virtual
  - python -m venv venv
  - source venv/bin/activate
- pip install -r requirements.txt


