# -*- coding: utf-8 -*-
"""
Script para la implementación del algoritmo de clasificación
"""

import random
import sys
import signal
import argparse
import pandas as pd
import numpy as np
import string
import pickle
import time
import json
import csv
import os
from colorama import Fore
# Sklearn
from sklearn.calibration import LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
# Nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
# Imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def signal_handler(sig, frame):
    """
    Función para manejar la señal SIGINT (Ctrl+C)
    :param sig: Señal
    :param frame: Frame
    """
    print("\nSaliendo del programa...")
    sys.exit(0)

def parse_args():
    """
    Función para parsear los argumentos de entrada
    """
    parse = argparse.ArgumentParser(description="Practica de algoritmos de clasificación de datos.")
    parse.add_argument("-m", "--mode", help="Modo de ejecución (train o test)", required=True)
    parse.add_argument("-f", "--file", help="Fichero csv (/Path_to_file)", required=True)
    parse.add_argument("-a", "--algorithm", help="Algoritmo a ejecutar (kNN, decision_tree o random_forest)", required=True)
    parse.add_argument("-p", "--prediction", help="Columna a predecir (Nombre de la columna)", required=True)
    parse.add_argument("-e", "--estimator", help="Estimador a utilizar para elegir el mejor modelo https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter", required=False, default=None)
    parse.add_argument("-c", "--cpu", help="Número de CPUs a utilizar [-1 para usar todos]", required=False, default=-1, type=int)
    parse.add_argument("-v", "--verbose", help="Muestra las metricas por la terminal", required=False, default=False, action="store_true")
    parse.add_argument("--debug", help="Modo debug [Muestra informacion extra del preprocesado y almacena el resultado del mismo en un .csv]", required=False, default=False, action="store_true")
    # Parseamos los argumentos
    args = parse.parse_args()
    
    # Leemos los parametros del JSON
    with open('clasificador.json') as json_file:
        config = json.load(json_file)
    
    # Juntamos todo en una variable
    for key, value in config.items():
        setattr(args, key, value)
    
    # Parseamos los argumentos
    return args	
    
def load_data(file):
    """
    Función para cargar los datos de un fichero csv
    :param file: Fichero csv
    :return: Datos del fichero
    """
    try:
        data = pd.read_csv(file, encoding='utf-8')
        print(Fore.GREEN+"Datos cargados con éxito"+Fore.RESET)
        return data
    except Exception as e:
        print(Fore.RED+"Error al cargar los datos"+Fore.RESET)
        print(e)
        sys.exit(1)

# =============================================================================
# FUNCIONES PARA CALCULAR MÉTRICAS
# =============================================================================

def calculate_fscore(y_true, y_pred):
    """
    Calcula el F1-score micro y macro de las predicciones.

    Parámetros:
        y_true: array-like con las etiquetas reales.
        y_pred: array-like con las etiquetas predichas.

    Retorna:
        tuple: (f1_micro, f1_macro)
            - f1_micro: F1-score calculado de forma micro (global entre todas las clases).
            - f1_macro: F1-score calculado de forma macro (media de F1 por clase).
    """
    # F1 micro: tiene en cuenta el total de verdaderos positivos, falsos negativos
    # y falsos positivos de todas las clases conjuntamente
    f1_micro = f1_score(y_true, y_pred, average='micro')

    # F1 macro: calcula el F1 de cada clase por separado y luego hace la media
    # sin tener en cuenta el número de muestras por clase
    f1_macro = f1_score(y_true, y_pred, average='macro')

    return f1_micro, f1_macro

def calculate_confusion_matrix(y_true, y_pred):
    """
    Calcula la matriz de confusión de las predicciones.

    Parámetros:
        y_true: array-like con las etiquetas reales.
        y_pred: array-like con las etiquetas predichas.

    Retorna:
        ndarray: Matriz de confusión donde la fila i y columna j indica
                 cuántas muestras de la clase i fueron predichas como clase j.
    """
    # Usamos la función de sklearn para calcular la matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    return cm

def calculate_classification_report(y_true, y_pred):
    """
    Genera un informe detallado de las métricas de clasificación (precisión,
    recall, f1-score y soporte) para cada clase.

    Parámetros:
        y_true: array-like con las etiquetas reales.
        y_pred: array-like con las etiquetas predichas.

    Retorna:
        str: Informe de clasificación en formato texto con precisión, recall,
             f1-score y soporte por clase.
    """
    # classification_report devuelve un string con todas las métricas por clase
    report = classification_report(y_true, y_pred)
    return report

# =============================================================================
# FUNCIONES PARA PREPROCESAR LOS DATOS
# =============================================================================

def select_features():
    """
    Separa las características del conjunto de datos en características numéricas,
    de texto y categóricas.

    Returns:
        numerical_feature (DataFrame): DataFrame con las características numéricas.
        text_feature (DataFrame): DataFrame con las características de texto.
        categorical_feature (DataFrame): DataFrame con las características categóricas.
    """
    try:
        # --- Características numéricas ---
        # Seleccionamos columnas de tipo entero o flotante
        numerical_feature = data.select_dtypes(include=['int64', 'float64'])
        # Eliminamos la columna objetivo si está entre las numéricas
        if args.prediction in numerical_feature.columns:
            numerical_feature = numerical_feature.drop(columns=[args.prediction])

        # --- Características categóricas ---
        # Seleccionamos columnas de tipo objeto (texto/cadenas)
        # Solo consideramos categóricas las que tienen pocos valores únicos
        categorical_feature = data.select_dtypes(include='object')
        categorical_feature = categorical_feature.loc[
            :, categorical_feature.nunique() <= args.preprocessing["unique_category_threshold"]
        ]

        # --- Características de texto ---
        # El resto de columnas de tipo objeto (las que tienen muchos valores únicos)
        # se consideran texto libre
        text_feature = data.select_dtypes(include='object').drop(columns=categorical_feature.columns)

        print(Fore.GREEN+"Datos separados con éxito"+Fore.RESET)

        if args.debug:
            print(Fore.MAGENTA+"> Columnas numéricas:\n"+Fore.RESET, numerical_feature.columns)
            print(Fore.MAGENTA+"> Columnas de texto:\n"+Fore.RESET, text_feature.columns)
            print(Fore.MAGENTA+"> Columnas categóricas:\n"+Fore.RESET, categorical_feature.columns)

        return numerical_feature, text_feature, categorical_feature

    except Exception as e:
        print(Fore.RED+"Error al separar los datos"+Fore.RESET)
        print(e)
        sys.exit(1)


def process_missing_values(numerical_feature, categorical_feature):
    """
    Procesa los valores faltantes (NaN) en los datos según la estrategia
    especificada en args.preprocessing["missing_values"].

    Estrategias disponibles:
        - "drop"  : Elimina las filas que contengan algún valor faltante.
        - "mean"  : Imputa los NaN de columnas numéricas con la media de la columna.
        - "median": Imputa los NaN de columnas numéricas con la mediana de la columna.
        - "mode"  : Imputa los NaN (numéricos y categóricos) con la moda de la columna.

    Parámetros:
        numerical_feature (DataFrame): DataFrame con las características numéricas.
        categorical_feature (DataFrame): DataFrame con las características categóricas.
    """
    global data

    try:
        # Obtenemos la estrategia configurada en el JSON
        strategy = args.preprocessing["missing_values"]

        if strategy == "drop":
            # Eliminamos todas las filas que tengan al menos un NaN
            filas_antes = len(data)
            data.dropna(inplace=True)
            filas_despues = len(data)
            print(Fore.GREEN
                  + f"Valores faltantes eliminados: {filas_antes - filas_despues} filas borradas"
                  + Fore.RESET)

        elif strategy == "mean":
            # Imputamos NaN numéricos con la media de cada columna
            for col in numerical_feature.columns:
                if data[col].isnull().any():
                    media = data[col].mean()
                    data[col].fillna(media, inplace=True)
            print(Fore.GREEN+"Valores faltantes numéricos imputados con la media"+Fore.RESET)

        elif strategy == "median":
            # Imputamos NaN numéricos con la mediana de cada columna
            for col in numerical_feature.columns:
                if data[col].isnull().any():
                    mediana = data[col].median()
                    data[col].fillna(mediana, inplace=True)
            print(Fore.GREEN+"Valores faltantes numéricos imputados con la mediana"+Fore.RESET)

        elif strategy == "mode":
            # Imputamos NaN tanto en numéricas como en categóricas con la moda
            for col in numerical_feature.columns:
                if data[col].isnull().any():
                    moda = data[col].mode()[0]  # mode() devuelve una Serie; cogemos el primer valor
                    data[col].fillna(moda, inplace=True)
            for col in categorical_feature.columns:
                if data[col].isnull().any():
                    moda = data[col].mode()[0]
                    data[col].fillna(moda, inplace=True)
            print(Fore.GREEN+"Valores faltantes imputados con la moda"+Fore.RESET)

        else:
            print(Fore.YELLOW+"Estrategia de valores faltantes no reconocida, no se aplica ninguna"+Fore.RESET)

    except Exception as e:
        print(Fore.RED+"Error al procesar los valores faltantes"+Fore.RESET)
        print(e)
        sys.exit(1)


def reescaler(numerical_feature):
    """
    Rescala las características numéricas del conjunto de datos usando el método
    especificado en args.preprocessing["scaler"].

    Métodos disponibles:
        - "minmax"    : Escala cada columna al rango [0, 1].
        - "standard"  : Estandariza cada columna (media 0, desviación típica 1).
        - "normalizer": Normaliza cada fila para que tenga norma unitaria (L2).
        - "maxabs"    : Escala cada columna por su valor absoluto máximo → rango [-1, 1].

    Parámetros:
        numerical_feature (DataFrame): DataFrame con las características numéricas.
    """
    global data

    try:
        # Comprobamos que haya columnas numéricas que escalar
        if numerical_feature.columns.size == 0:
            print(Fore.YELLOW+"No hay columnas numéricas para reescalar"+Fore.RESET)
            return

        # Obtenemos el método de escalado configurado
        scaler_method = args.preprocessing["scaler"]

        # Seleccionamos el escalador según la configuración
        if scaler_method == "minmax":
            # MinMaxScaler: transforma cada valor a x' = (x - min) / (max - min)
            scaler = MinMaxScaler()
        elif scaler_method == "standard":
            # StandardScaler: transforma a x' = (x - media) / desv_tipica
            scaler = StandardScaler()
        elif scaler_method == "normalizer":
            # Normalizer: normaliza cada fila (muestra) por su norma L2
            scaler = Normalizer()
        elif scaler_method == "maxabs":
            # MaxAbsScaler: divide cada valor por el máximo absoluto de la columna
            scaler = MaxAbsScaler()
        else:
            print(Fore.YELLOW+"Método de escalado no reconocido, no se aplica ninguno"+Fore.RESET)
            return

        # Aplicamos el escalado y actualizamos las columnas en el DataFrame principal
        data[numerical_feature.columns] = scaler.fit_transform(data[numerical_feature.columns])
        print(Fore.GREEN+f"Datos numéricos reescalados con éxito usando '{scaler_method}'"+Fore.RESET)

    except Exception as e:
        print(Fore.RED+"Error al reescalar los datos"+Fore.RESET)
        print(e)
        sys.exit(1)


def cat2num(categorical_feature):
    """
    Convierte las columnas categóricas (texto) a valores numéricos usando LabelEncoder.
    LabelEncoder asigna un entero único a cada categoría distinta.

    Parámetros:
        categorical_feature (DataFrame): DataFrame con las columnas categóricas.
    """
    # Usamos la variable global 'data' (DataFrame principal)
    global data

    try:
        # Comprobamos si hay columnas categóricas
        if categorical_feature.columns.size > 0:

            # Creamos un objeto LabelEncoder para convertir texto a números
            le = LabelEncoder()

            # Recorremos cada columna categórica
            for col in categorical_feature.columns:
                # Convertimos la columna a string (por si hay NaN u otros tipos)
                # y aplicamos fit_transform:
                # - fit: aprende las categorías únicas
                # - transform: reemplaza cada categoría por su índice numérico
                data[col] = le.fit_transform(data[col].astype(str))

            print(Fore.GREEN+"Variables categóricas convertidas a numéricas"+Fore.RESET)

        else:
            # Si no hay columnas categóricas, no hacemos nada
            print(Fore.YELLOW+"No hay variables categóricas"+Fore.RESET)

    except Exception as e:
        print(Fore.RED+"Error en cat2num"+Fore.RESET)
        print(e)
        sys.exit(1)


def simplify_text(text_feature):
    """
    Simplifica el texto de las columnas de texto libre aplicando los siguientes pasos:
        1. Conversión a minúsculas.
        2. Eliminación de signos de puntuación.
        3. Tokenización (división en palabras individuales).
        4. Eliminación de stopwords (palabras vacías sin significado semántico).
        5. Stemming (reducción de cada palabra a su raíz morfológica).
        6. Reordenación alfabética de los tokens (para normalizar el orden).

    Parámetros:
        text_feature (DataFrame): DataFrame con las columnas de texto a simplificar.
    """
    global data

    try:
        # Comprobamos si hay columnas de texto que procesar
        if text_feature.columns.size == 0:
            print(Fore.YELLOW+"No hay columnas de texto que simplificar"+Fore.RESET)
            return

        # Obtenemos las stopwords en inglés (cambiar a 'spanish' si el corpus es en español)
        stop_words = set(stopwords.words('english'))

        # Creamos el stemmer de Porter (reduce palabras a su raíz, ej. "running" → "run")
        stemmer = PorterStemmer()

        # Conjunto de signos de puntuación que queremos eliminar
        puntuacion = set(string.punctuation)

        # Función auxiliar que aplica todos los pasos a una cadena de texto
        def pipeline_texto(texto):
            # 1. Convertir a minúsculas para uniformizar el texto
            texto = str(texto).lower()

            # 2. Eliminar signos de puntuación carácter a carácter
            texto = ''.join(ch for ch in texto if ch not in puntuacion)

            # 3. Tokenizar: dividir la cadena en una lista de palabras
            tokens = word_tokenize(texto)

            # 4. Eliminar stopwords (palabras como "the", "is", "at", etc.)
            tokens = [t for t in tokens if t not in stop_words]

            # 5. Aplicar stemming a cada token restante
            tokens = [stemmer.stem(t) for t in tokens]

            # 6. Ordenar alfabéticamente los tokens para normalizar el orden
            tokens.sort()

            # Reunimos los tokens en una cadena de texto limpia
            return ' '.join(tokens)

        # Aplicamos el pipeline a cada columna de texto
        for col in text_feature.columns:
            data[col] = data[col].apply(pipeline_texto)

        print(Fore.GREEN+"Texto simplificado con éxito"+Fore.RESET)

    except Exception as e:
        print(Fore.RED+"Error al simplificar el texto"+Fore.RESET)
        print(e)
        sys.exit(1)


def process_text(text_feature):
    """
    Procesa las características de texto utilizando técnicas de vectorización:
        - "tf-idf": Term Frequency - Inverse Document Frequency.
        - "bow"   : Bag of Words (frecuencia de aparición de cada término).

    Parámetros:
        text_feature (DataFrame): DataFrame con las columnas de texto a vectorizar.
    """
    global data
    try:
        if text_feature.columns.size > 0:
            if args.preprocessing["text_process"] == "tf-idf":
                # TF-IDF pondera cada término por su frecuencia en el documento
                # y penaliza los que aparecen en muchos documentos (poca información)
                tfidf_vectorizer = TfidfVectorizer()
                # Concatenamos todas las columnas de texto en una sola cadena por fila
                text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
                # Generamos la matriz TF-IDF
                tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
                # Convertimos la matriz dispersa en un DataFrame con nombres de columna
                text_features_df = pd.DataFrame(
                    tfidf_matrix.toarray(),
                    columns=tfidf_vectorizer.get_feature_names_out()
                )
                # Añadimos las nuevas columnas al DataFrame principal
                data = pd.concat([data, text_features_df], axis=1)
                # Eliminamos las columnas de texto originales (ya vectorizadas)
                data.drop(text_feature.columns, axis=1, inplace=True)
                print(Fore.GREEN+"Texto tratado con éxito usando TF-IDF"+Fore.RESET)

            elif args.preprocessing["text_process"] == "bow":
                # BOW cuenta simplemente cuántas veces aparece cada término
                bow_vectorizer = CountVectorizer()
                text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
                bow_matrix = bow_vectorizer.fit_transform(text_data)
                text_features_df = pd.DataFrame(
                    bow_matrix.toarray(),
                    columns=bow_vectorizer.get_feature_names_out()
                )
                data = pd.concat([data, text_features_df], axis=1)
                # Eliminamos las columnas de texto originales
                data.drop(text_feature.columns, axis=1, inplace=True)
                print(Fore.GREEN+"Texto tratado con éxito usando BOW"+Fore.RESET)

            else:
                print(Fore.YELLOW+"No se están tratando los textos"+Fore.RESET)
        else:
            print(Fore.YELLOW+"No se han encontrado columnas de texto a procesar"+Fore.RESET)

    except Exception as e:
        print(Fore.RED+"Error al tratar el texto"+Fore.RESET)
        print(e)
        sys.exit(1)


def over_under_sampling():
    """
    Aplica oversampling o undersampling sobre los datos de entrenamiento para
    equilibrar las clases de la variable objetivo, según la estrategia
    especificada en args.preprocessing["sampling"].

    Estrategias disponibles:
        - "oversampling" : Replica aleatoriamente muestras de las clases minoritarias
                           hasta igualar la clase mayoritaria (RandomOverSampler).
        - "undersampling": Elimina aleatoriamente muestras de las clases mayoritarias
                           hasta igualar la clase minoritaria (RandomUnderSampler).
        - Cualquier otro valor: No se aplica ningún muestreo.
    """
    global data

    try:
        # Obtenemos la estrategia de muestreo de la configuración
        sampling_strategy = args.preprocessing["sampling"]

        if sampling_strategy == "oversampling":
            # Separamos las características (X) de la etiqueta objetivo (y)
            X = data.drop(columns=[args.prediction])
            y = data[args.prediction]

            # RandomOverSampler duplica muestras de clases minoritarias aleatoriamente
            ros = RandomOverSampler(random_state=42)
            X_resampled, y_resampled = ros.fit_resample(X, y)

            # Reconstruimos el DataFrame con los datos balanceados
            data = pd.DataFrame(X_resampled, columns=X.columns)
            data[args.prediction] = y_resampled

            print(Fore.GREEN
                  + f"Oversampling aplicado: {len(data)} muestras totales"
                  + Fore.RESET)

        elif sampling_strategy == "undersampling":
            # Separamos características y etiqueta
            X = data.drop(columns=[args.prediction])
            y = data[args.prediction]

            # RandomUnderSampler elimina muestras de clases mayoritarias aleatoriamente
            rus = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = rus.fit_resample(X, y)

            # Reconstruimos el DataFrame con los datos reducidos
            data = pd.DataFrame(X_resampled, columns=X.columns)
            data[args.prediction] = y_resampled

            print(Fore.GREEN
                  + f"Undersampling aplicado: {len(data)} muestras totales"
                  + Fore.RESET)

        else:
            # No aplicamos ningún tipo de muestreo
            print(Fore.YELLOW+"No se aplica ningún tipo de muestreo"+Fore.RESET)

    except Exception as e:
        print(Fore.RED+"Error al realizar el oversampling/undersampling"+Fore.RESET)
        print(e)
        sys.exit(1)


def drop_features():
    """
    Elimina del DataFrame las columnas especificadas en
    args.preprocessing["drop_features"].
    """
    global data
    try:
        data = data.drop(columns=args.preprocessing["drop_features"])
        print(Fore.GREEN+"Columnas eliminadas con éxito"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al eliminar columnas"+Fore.RESET)
        print(e)
        sys.exit(1)


def preprocesar_datos():
    """
    Función principal de preprocesamiento. Ejecuta los pasos en orden:
        1. Separar los datos por tipos (numérico, texto, categórico).
        2. Simplificar el texto (minúsculas, stopwords, stemming...).
        3. Convertir categóricas a numéricas (LabelEncoder).
        4. Tratar valores faltantes (eliminar o imputar).
        5. Reescalar variables numéricas.
        6. Vectorizar texto (TF-IDF o BOW).
        7. Balancear clases (over/undersampling).
        8. Eliminar columnas innecesarias.

    Retorna:
        DataFrame preprocesado listo para entrenar o predecir.
    """
    # 1. Separamos los datos por tipos
    numerical_feature, text_feature, categorical_feature = select_features()

    # 2. Simplificamos el texto (normalización, stopwords, stemming, orden)
    simplify_text(text_feature)

    # 3. Convertimos categóricas a numéricas
    cat2num(categorical_feature)

    # 4. Tratamos valores faltantes
    process_missing_values(numerical_feature, categorical_feature)

    # 5. Reescalamos los datos numéricos
    reescaler(numerical_feature)

    # 6. Vectorizamos el texto (TF-IDF o BOW)
    process_text(text_feature)

    # 7. Balanceamos las clases
    over_under_sampling()

    # 8. Eliminamos columnas no necesarias
    drop_features()

    return data

# =============================================================================
# FUNCIONES PARA ENTRENAR UN MODELO
# =============================================================================

def divide_data():
    """
    Divide los datos en conjuntos de entrenamiento y desarrollo (validación).

    Utiliza train_test_split de sklearn con la proporción configurada en
    args.preprocessing["test_size"] y semilla fija para reproducibilidad.

    Retorna:
        x_train (DataFrame): Características de entrenamiento.
        x_dev   (DataFrame): Características de desarrollo/validación.
        y_train (Series)   : Etiquetas de entrenamiento.
        y_dev   (Series)   : Etiquetas de desarrollo/validación.
    """
    # Separamos la columna objetivo del resto de características
    X = data.drop(columns=[args.prediction])   # Matriz de características
    y = data[args.prediction]                  # Vector de etiquetas

    # Dividimos en train y dev con estratificación para mantener la proporción de clases
    x_train, x_dev, y_train, y_dev = train_test_split(
        X, y,
        test_size=args.preprocessing.get("test_size", 0.2),  # 20% para dev por defecto
        random_state=42,       # Semilla fija para reproducibilidad
        stratify=y             # Mantiene la proporción de clases en ambos subconjuntos
    )

    if args.debug:
        print(Fore.MAGENTA
              + f"> Train: {len(x_train)} muestras | Dev: {len(x_dev)} muestras"
              + Fore.RESET)

    return x_train, x_dev, y_train, y_dev


def save_model(gs):
    """
    Guarda el modelo entrenado (objeto GridSearchCV) y los resultados del
    barrido de hiperparámetros en archivos.

    Archivos generados:
        - output/modelo.pkl : Modelo serializado con pickle.
        - output/modelo.csv : Resultados del GridSearchCV (parámetros y puntuación).

    Parámetros:
        gs: Objeto GridSearchCV ya entrenado.
    """
    try:
        # Guardamos el modelo entrenado en formato binario (pickle)
        with open('output/modelo.pkl', 'wb') as file:
            pickle.dump(gs, file)
            print(Fore.CYAN+"Modelo guardado con éxito"+Fore.RESET)

        # Guardamos los resultados del barrido de hiperparámetros en CSV
        with open('output/modelo.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['Params', 'Score'])
            for params, score in zip(gs.cv_results_['params'], gs.cv_results_['mean_test_score']):
                writer.writerow([params, score])
    except Exception as e:
        print(Fore.RED+"Error al guardar el modelo"+Fore.RESET)
        print(e)


def mostrar_resultados(gs, x_dev, y_dev):
    """
    Muestra por pantalla (si --verbose está activo) las métricas del modelo
    evaluado sobre el conjunto de desarrollo.

    Parámetros:
        gs   : Objeto GridSearchCV ya entrenado.
        x_dev: Características del conjunto de desarrollo.
        y_dev: Etiquetas del conjunto de desarrollo.
    """
    if args.verbose:
        print(Fore.MAGENTA+"> Mejores parametros:\n"+Fore.RESET, gs.best_params_)
        print(Fore.MAGENTA+"> Mejor puntuacion:\n"+Fore.RESET, gs.best_score_)
        print(Fore.MAGENTA+"> F1-score micro:\n"+Fore.RESET,
              calculate_fscore(y_dev, gs.predict(x_dev))[0])
        print(Fore.MAGENTA+"> F1-score macro:\n"+Fore.RESET,
              calculate_fscore(y_dev, gs.predict(x_dev))[1])
        print(Fore.MAGENTA+"> Informe de clasificación:\n"+Fore.RESET,
              calculate_classification_report(y_dev, gs.predict(x_dev)))
        print(Fore.MAGENTA+"> Matriz de confusión:\n"+Fore.RESET,
              calculate_confusion_matrix(y_dev, gs.predict(x_dev)))


def kNN():
    """
    Implementa el algoritmo k-Nearest Neighbors con búsqueda de hiperparámetros
    mediante GridSearchCV.

    Los hiperparámetros a explorar se leen de args.kNN (definidos en el JSON).
    Ejemplo de configuración en JSON:
        "kNN": {
            "n_neighbors": [3, 5, 7, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]
        }
    """
    # Dividimos los datos en entrenamiento y desarrollo
    x_train, x_dev, y_train, y_dev = divide_data()

    # Barrido de hiperparámetros con validación cruzada de 5 particiones
    with tqdm(total=100, desc='Procesando kNN', unit='iter', leave=True) as pbar:
        gs = GridSearchCV(
            KNeighborsClassifier(),  # Clasificador base
            args.kNN,                # Grid de hiperparámetros del JSON
            cv=5,                    # 5-fold cross-validation
            n_jobs=args.cpu,         # Número de CPUs (-1 = todos)
            scoring=args.estimator   # Métrica de evaluación (None = accuracy por defecto)
        )
        start_time = time.time()
        gs.fit(x_train, y_train)     # Entrenamos con los datos de train
        end_time = time.time()
        # Animación de la barra de progreso (el entrenamiento ya terminó)
        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))
            pbar.update(random.random() * 2)
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)

    execution_time = end_time - start_time
    print("Tiempo de ejecución:" + Fore.MAGENTA, execution_time, Fore.RESET + "segundos")

    # Mostramos métricas si --verbose está activo
    mostrar_resultados(gs, x_dev, y_dev)

    # Guardamos el modelo en disco
    save_model(gs)


def decision_tree():
    """
    Implementa el algoritmo de Árbol de Decisión con búsqueda de hiperparámetros
    mediante GridSearchCV.

    Los hiperparámetros a explorar se leen de args.decision_tree (definidos en el JSON).
    Ejemplo de configuración en JSON:
        "decision_tree": {
            "max_depth": [null, 5, 10, 20],
            "criterion": ["gini", "entropy"],
            "min_samples_split": [2, 5, 10]
        }
    """
    # Dividimos los datos en entrenamiento y desarrollo
    x_train, x_dev, y_train, y_dev = divide_data()

    # Barrido de hiperparámetros con validación cruzada de 5 particiones
    with tqdm(total=100, desc='Procesando decision tree', unit='iter', leave=True) as pbar:
        gs = GridSearchCV(
            DecisionTreeClassifier(random_state=42),  # Clasificador base con semilla fija
            args.decision_tree,                       # Grid de hiperparámetros del JSON
            cv=5,                                     # 5-fold cross-validation
            n_jobs=args.cpu,                          # Número de CPUs
            scoring=args.estimator                    # Métrica de evaluación
        )
        start_time = time.time()
        gs.fit(x_train, y_train)                      # Entrenamos con los datos de train
        end_time = time.time()
        # Animación de la barra de progreso
        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))
            pbar.update(random.random() * 2)
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)

    execution_time = end_time - start_time
    print("Tiempo de ejecución:" + Fore.MAGENTA, execution_time, Fore.RESET + "segundos")

    # Mostramos métricas si --verbose está activo
    mostrar_resultados(gs, x_dev, y_dev)

    # Guardamos el modelo en disco
    save_model(gs)


def random_forest():
    """
    Implementa el algoritmo Random Forest con búsqueda de hiperparámetros
    mediante GridSearchCV.

    Random Forest es un ensemble de árboles de decisión entrenados sobre
    subconjuntos aleatorios de datos y características. La predicción final
    se obtiene por votación mayoritaria de todos los árboles.

    Los hiperparámetros a explorar se leen de args.random_forest (definidos en el JSON).
    Ejemplo de configuración en JSON:
        "random_forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [null, 5, 10],
            "criterion": ["gini", "entropy"],
            "min_samples_split": [2, 5]
        }
    """
    # Dividimos los datos en entrenamiento y desarrollo
    x_train, x_dev, y_train, y_dev = divide_data()

    # Barrido de hiperparámetros con validación cruzada de 5 particiones
    with tqdm(total=100, desc='Procesando random forest', unit='iter', leave=True) as pbar:
        gs = GridSearchCV(
            RandomForestClassifier(random_state=42),  # Clasificador base con semilla fija
            args.random_forest,                       # Grid de hiperparámetros del JSON
            cv=5,                                     # 5-fold cross-validation
            n_jobs=args.cpu,                          # Número de CPUs
            scoring=args.estimator                    # Métrica de evaluación
        )
        start_time = time.time()
        gs.fit(x_train, y_train)                      # Entrenamos con los datos de train
        end_time = time.time()
        # Animación de la barra de progreso
        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))
            pbar.update(random.random() * 2)
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)

    execution_time = end_time - start_time
    print("Tiempo de ejecución:" + Fore.MAGENTA, execution_time, Fore.RESET + "segundos")

    # Mostramos métricas si --verbose está activo
    mostrar_resultados(gs, x_dev, y_dev)

    # Guardamos el modelo en disco
    save_model(gs)

# =============================================================================
# FUNCIONES PARA PREDECIR CON UN MODELO
# =============================================================================

def load_model():
    """
    Carga el modelo previamente entrenado desde 'output/modelo.pkl'.

    Retorna:
        model: Objeto GridSearchCV cargado desde el archivo.
    """
    try:
        with open('output/modelo.pkl', 'rb') as file:
            model = pickle.load(file)
            print(Fore.GREEN+"Modelo cargado con éxito"+Fore.RESET)
            return model
    except Exception as e:
        print(Fore.RED+"Error al cargar el modelo"+Fore.RESET)
        print(e)
        sys.exit(1)


def predict():
    """
    Realiza predicciones sobre los datos cargados usando el modelo entrenado,
    añade la columna de predicciones al DataFrame y la guarda en CSV.
    """
    global data

    # Eliminamos la columna objetivo si está presente (en modo test puede no estarlo)
    if args.prediction in data.columns:
        X = data.drop(columns=[args.prediction])
    else:
        X = data

    # Realizamos la predicción con el mejor estimador encontrado por GridSearchCV
    prediction = model.predict(X)

    # Añadimos la predicción como nueva columna al DataFrame
    data = pd.concat(
        [data, pd.DataFrame(prediction, columns=[args.prediction])],
        axis=1
    )

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    # Fijamos la semilla para reproducibilidad
    np.random.seed(42)
    print("=== Clasificador ===")

    # Manejamos la señal SIGINT (Ctrl+C) para salir limpiamente
    signal.signal(signal.SIGINT, signal_handler)

    # Parseamos los argumentos de línea de comandos y el JSON de configuración
    args = parse_args()

    # Si la carpeta output no existe, la creamos
    print("\n- Creando carpeta output...")
    try:
        os.makedirs('output')
        print(Fore.GREEN+"Carpeta output creada con éxito"+Fore.RESET)
    except FileExistsError:
        print(Fore.GREEN+"La carpeta output ya existe"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al crear la carpeta output"+Fore.RESET)
        print(e)
        sys.exit(1)

    # Cargamos los datos del fichero CSV indicado
    print("\n- Cargando datos...")
    data = load_data(args.file)

    # Descargamos los recursos necesarios de NLTK
    print("\n- Descargando diccionarios...")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

    # Preprocesamos los datos
    print("\n- Preprocesando datos...")
    preprocesar_datos()

    # Si está en modo debug, guardamos el CSV preprocesado para inspección
    if args.debug:
        try:
            print("\n- Guardando datos preprocesados...")
            data.to_csv('output/data-processed.csv', index=False)
            print(Fore.GREEN+"Datos preprocesados guardados con éxito"+Fore.RESET)
        except Exception as e:
            print(Fore.RED+"Error al guardar los datos preprocesados"+Fore.RESET)

    # ------------------------------------------------------------------
    # MODO ENTRENAMIENTO
    # ------------------------------------------------------------------
    if args.mode == "train":
        print("\n- Ejecutando algoritmo...")

        if args.algorithm == "kNN":
            try:
                kNN()
                print(Fore.GREEN+"Algoritmo kNN ejecutado con éxito"+Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)

        elif args.algorithm == "decision_tree":
            try:
                decision_tree()
                print(Fore.GREEN+"Algoritmo árbol de decisión ejecutado con éxito"+Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)

        elif args.algorithm == "random_forest":
            try:
                random_forest()
                print(Fore.GREEN+"Algoritmo random forest ejecutado con éxito"+Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)

        else:
            print(Fore.RED+"Algoritmo no soportado"+Fore.RESET)
            sys.exit(1)

    # ------------------------------------------------------------------
    # MODO PREDICCIÓN
    # ------------------------------------------------------------------
    elif args.mode == "test":
        # Cargamos el modelo previamente entrenado
        print("\n- Cargando modelo...")
        model = load_model()

        # Realizamos la predicción
        print("\n- Prediciendo...")
        try:
            predict()
            print(Fore.GREEN+"Predicción realizada con éxito"+Fore.RESET)
            # Guardamos el DataFrame con las predicciones
            data.to_csv('output/data-prediction.csv', index=False)
            print(Fore.GREEN+"Predicción guardada con éxito"+Fore.RESET)
            sys.exit(0)
        except Exception as e:
            print(e)
            sys.exit(1)

    else:
        print(Fore.RED+"Modo no soportado"+Fore.RESET)
        sys.exit(1)
