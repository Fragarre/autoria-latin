import os
import zipfile
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# --- Función para extraer el archivo ZIP --- #
def unzip_data(zip_file, extract_path='./data/'):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    st.success(f"¡Datos extraídos correctamente a {extract_path}!")

# --- Función para cargar los textos y generar el modelo --- #
def load_and_train_model(ngram_min, ngram_max):
    corpus_path = './data/'
    texts = []
    labels = []
    filenames = []

    for author_folder in os.listdir(corpus_path):
        author_path = os.path.join(corpus_path, author_folder)
        if os.path.isdir(author_path):
            for filename in os.listdir(author_path):
                if filename.endswith('.txt'):
                    with open(os.path.join(author_path, filename), 'r', encoding='utf-8') as f:
                        texts.append(f.read())
                        labels.append(author_folder)
                        filenames.append(f"{author_folder}/{filename[:-4]}")  # Nombre autor/archivo

    # Vectorización de textos
    # vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))  # Usar los ngramas del usuario
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(ngram_min, ngram_max)) # Cambia aquí
    X = vectorizer.fit_transform(texts)

    # --- Reducción de dimensión --- #
    max_components = min(X.shape[0], X.shape[1]) - 1
    n_components = min(50, max_components)

    if n_components < 2:
        st.warning("Muy pocos datos para aplicar SVD o UMAP. Intenta subir más textos.")
    else:
        # Convertir a matriz densa antes de aplicar SVD
        X_dense = X.toarray()

        svd = TruncatedSVD(n_components=n_components, random_state=42)
        X_reduced = svd.fit_transform(X_dense)

        # --- Codificación de etiquetas y entrenamiento --- #
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)

        clf = NearestCentroid()
        clf.fit(X_reduced, labels_encoded)

    return clf, vectorizer, svd, texts, labels, filenames, X_reduced, le

# --- Función para realizar predicciones con el modelo cargado --- #
def predict_with_model(clf, vectorizer, svd, le, test_texts, test_filenames):
    # Vectorización de los textos de prueba
    X_test = vectorizer.transform(test_texts)

    # Convertir a matriz densa antes de reducir la dimensionalidad
    X_test_dense = X_test.toarray()

    # Reducción de dimensión de los datos de prueba
    X_test_reduced = svd.transform(X_test_dense)

    # Predicción
    y_pred = clf.predict(X_test_reduced)
    
    # Obtener las probabilidades de cada clase (autor)
    y_prob = clf.predict_proba(X_test_reduced)

    # Decodificar las predicciones
    pred_labels = le.inverse_transform(y_pred)

    # Crear un DataFrame con los resultados
    df_results = pd.DataFrame(y_prob, columns=le.classes_)
    
    # Limitar las probabilidades a 3 decimales
    # df_results = df_results.round(3)
    df_results = df_results.applymap(lambda x: f"{round(x * 100, 2)}%")

    # Incluir los nombres de las obras en la primera columna
    df_results['Obra'] = test_filenames
    # df_results['Autor predicho'] = pred_labels

    # Reorganizar para que la columna 'Obra' esté primero
    # df_results = df_results[['Obra', 'Autor predicho'] + list(le.classes_)]
    df_results = df_results[['Obra'] + list(le.classes_)]
    return df_results, X_test_reduced, y_pred

# --- Interfaz de usuario en Streamlit --- #
st.title("Autoría de textos latinos")
st.write("Sube dos archivos .zip: uno con los datos de entrenamiento (data.zip) y otro con los datos de prueba (test.zip).")

uploaded_train_zip = st.sidebar.file_uploader("Sube el archivo `data.zip` para entrenamiento", type=["zip"])
uploaded_test_zip = st.sidebar.file_uploader("Sube el archivo `test.zip` para prueba", type=["zip"])

# --- Sidebar con instrucciones y opciones de carga --- #
st.sidebar.markdown("""### Instrucciones
1. Carga un archivo `data.zip` con los textos de entrenamiento (con carpetas por autor).
2. Carga un archivo `test.zip` con los textos de prueba (solo los textos a predecir).
""")
ngram_min = st.sidebar.number_input("n-grama mínimo", min_value=1, max_value=10, value=2)
ngram_max = st.sidebar.number_input("n-grama máximo", min_value=1, max_value=10, value=4)



if uploaded_train_zip is not None and uploaded_test_zip is not None:
    # Extraer y cargar los datos de entrenamiento
    unzip_data(uploaded_train_zip, extract_path='./data/')
    clf, vectorizer, svd, texts, labels, filenames, X_reduced, le = load_and_train_model(ngram_min, ngram_max)

    # Extraer y cargar los datos de prueba
    unzip_data(uploaded_test_zip, extract_path='./test_data/')
    test_texts = []
    test_filenames = []

    for filename in os.listdir('./test_data/'):
        if filename.endswith('.txt'):
            with open(os.path.join('./test_data/', filename), 'r', encoding='utf-8') as f:
                test_texts.append(f.read())
                test_filenames.append(filename[:-4])  # Nombre del archivo sin extensión

    # Realizar predicciones
    df_results, X_test_reduced, y_pred = predict_with_model(clf, vectorizer, svd, le, test_texts, test_filenames)
    # Mostrar la tabla de resultados
    st.subheader("Resultados de probabilidades de autoría")
    st.dataframe(df_results)

import base64
import os

def get_readme_download_link(readme_path="README.md"):
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:text/markdown;base64,{b64}" download="README.md" target="_blank">📖 Descargar el fichero README.md</a>'
    return href

st.markdown(get_readme_download_link(), unsafe_allow_html=True)

