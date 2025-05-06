import os
import zipfile
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import base64
import shutil

# --- Borrar automáticamente ./data al iniciar la app --- #
def clear_data_folder():
    data_path = './data/'
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.makedirs(data_path)

def clear_test_folder():
    data_path = './test_data/'
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.makedirs(data_path)

clear_data_folder()
clear_test_folder()

# --- Función para extraer el archivo ZIP --- #
def unzip_data(zip_file, extract_path='./data/'):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    st.success(f"¡Datos extraídos correctamente a {extract_path}! (Espera mientras el estado sea 'RUNNING...)")

# --- Función para cargar los textos y generar el modelo --- #
def load_and_train_model(ngram_min, ngram_max):
    corpus_path = './data/'
    texts = []
    labels = []
    filenames = []

    for filename in os.listdir(corpus_path):
        if filename.endswith('.txt'):
            with open(os.path.join(corpus_path, filename), 'r', encoding='utf-8') as f:
                texts.append(f.read())
                author = filename.split('_')[0]  # Autor antes del primer '_'
                labels.append(author)
                filenames.append(filename[:-4])  # sin .txt

    # Vectorización de textos
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(ngram_min, ngram_max))
    X = vectorizer.fit_transform(texts)

    # --- Reducción de dimensión --- #
    max_components = min(X.shape[0], X.shape[1]) - 1
    n_components = min(50, max_components)

    if n_components < 2:
        st.warning("Muy pocos datos para aplicar SVD. Intenta subir más textos.")
    else:
        X_dense = X.toarray()
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        X_reduced = svd.fit_transform(X_dense)

        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)

        clf = NearestCentroid()
        clf.fit(X_reduced, labels_encoded)

    return clf, vectorizer, svd, texts, labels, filenames, X_reduced, le

# --- Función para realizar predicciones con el modelo cargado --- #
def predict_with_model(clf, vectorizer, svd, le, test_texts, test_filenames):
    X_test = vectorizer.transform(test_texts)
    X_test_dense = X_test.toarray()
    X_test_reduced = svd.transform(X_test_dense)

    y_pred = clf.predict(X_test_reduced)
    y_prob = clf.predict_proba(X_test_reduced)

    pred_labels = le.inverse_transform(y_pred)
    df_results = pd.DataFrame(y_prob, columns=le.classes_)
    df_results = df_results.applymap(lambda x: f"{round(x * 100, 2)}%")
    df_results['Obra'] = test_filenames
    df_results = df_results[['Obra'] + list(le.classes_)]

    return df_results, X_test_reduced, y_pred

# --- Interfaz de usuario en Streamlit --- #
st.title("Autoría de textos latinos")
st.write("Sube dos archivos .zip: uno con los datos de entrenamiento (`data.zip`) y otro con los textos a testear (`test.zip`).")

uploaded_train_zip = st.sidebar.file_uploader("Sube el archivo `data.zip`", type=["zip"])
uploaded_test_zip = st.sidebar.file_uploader("Sube el archivo `test.zip`", type=["zip"])

st.sidebar.markdown("""### Instrucciones
1. `data.zip`: textos de entrenamiento en formato `Autor_Obra.txt`
2. `test.zip`: textos a predecir, cualquier nombre `.txt`
""")

ngram_min = st.sidebar.number_input("n-grama mínimo", min_value=1, max_value=10, value=2)
ngram_max = st.sidebar.number_input("n-grama máximo", min_value=1, max_value=10, value=4)

if uploaded_train_zip is not None and uploaded_test_zip is not None:
    unzip_data(uploaded_train_zip, extract_path='./data/')
    clf, vectorizer, svd, texts, labels, filenames, X_reduced, le = load_and_train_model(ngram_min, ngram_max)

    unzip_data(uploaded_test_zip, extract_path='./test_data/')
    test_texts = []
    test_filenames = []

    for filename in os.listdir('./test_data/'):
        if filename.endswith('.txt'):
            with open(os.path.join('./test_data/', filename), 'r', encoding='utf-8') as f:
                test_texts.append(f.read())
                test_filenames.append(filename)  # Nombre completo con .txt

    df_results, X_test_reduced, y_pred = predict_with_model(clf, vectorizer, svd, le, test_texts, test_filenames)

    st.subheader("Resultados de probabilidades de autoría")
    st.dataframe(df_results)

# --- Opción de descarga del README --- #

# st.markdown(get_readme_download_link(), unsafe_allow_html=True)
