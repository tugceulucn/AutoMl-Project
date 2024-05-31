import zipfile
import json, time
import inspect
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yaml, joblib, pickle

from bokeh.plotting import figure
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, \
    RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_squared_error, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from streamlit_card import card
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
from warnings import filterwarnings
from sklearn.exceptions import ConvergenceWarning
import time

filterwarnings('ignore', category=ConvergenceWarning)

#GENEL FONKSİYONLAR
def handle_file_upload(uploaded_file):
    try:
        if uploaded_file is None:
            st.error('Lütfen bir veri seti yükleyin. Veri setinizi eğitim ve test için hazırladıysanız, bir sonraki adıma geçin.', icon="🚨")
        else:
            if uploaded_file.type == 'text/csv':  # CSV dosyası yüklenirse
                df = pd.read_csv(uploaded_file)
                st.success("CSV Data loaded successfully. Let's continue with Machine Learning!", icon="✅")
            elif uploaded_file.type == 'text/yaml':  # YAML dosyası yüklenirse
                    yaml_verisi = yaml.safe_load(uploaded_file)
                    # YAML verisini DataFrame'e dönüştür
                    df = pd.DataFrame(yaml_verisi)
                    # CSV dosyasına kaydet
                    df.to_csv("veri.csv", index=False)
                    # CSV dosyasını tekrar yükle
                    df = pd.read_csv("veri.csv")
                    st.success("YAML Data loaded successfully.Let's continue with Machine Learning!", icon="✅")
            elif uploaded_file.type == 'application/json':  # JSON dosyası yüklenirse
                    json_verisi = json.load(uploaded_file)
                    # JSON verisini DataFrame'e dönüştür
                    df = pd.DataFrame(json_verisi)
                    # CSV dosyasına kaydet
                    df.to_csv("veri.csv", index=False)
                    # CSV dosyasını tekrar yükle
                    df = pd.read_csv("veri.csv")
                    st.success("JSON Data loaded successfully.Let's continue with Machine Learning!", icon="✅")
            elif uploaded_file.type in ['application/zip', 'application/x-zip-compressed']:  # Zip dosyası yüklenirse (resim klasörü)
                    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                        zip_ref.extractall("extracted_images")
                    st.success("ZIP/Image folder has been successfully uploaded and opened.Let's continue with Machine Learning!", icon="✅")
            else:
                st.write("The file format is not supported.")
                
            if df.empty:
                    st.error('The loaded data set is empty.', icon="🚨")
            
            return df
        
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        st.error("No columns to parse from file. Please check the file content.", icon="🚨")
    except Exception as e:
        st.error(f"Error reading the file: {e}", icon="🚨")   

def data_preprocessing(df):
    try:
        # Yinelenen satırları kaldırma
        df.drop_duplicates(inplace=True)
        # Label Encoding işlemi burada gerçekleşecek
        label_encoder = LabelEncoder()
        string_columns = df.select_dtypes(include=['object']).columns
        df[string_columns] = df[string_columns].apply(label_encoder.fit_transform)
        # Integer değerlere ortalama ile eksik değerleri doldurma
        integer_columns = df.select_dtypes(include=['int', 'float']).columns
        df[integer_columns] = df[integer_columns].fillna(df[integer_columns].mean())
        # String değerlere "Bilinmiyor" ile eksik değerleri doldurma
        string_columns = df.select_dtypes(include=['object']).columns
        df[string_columns] = df[string_columns].fillna("Bilinmiyor")
        
        return df, label_encoder
    
    except Exception as e:
        st.write(f"Hata: {e}")
        return None

#AUTOML FONKSİYONLARI
def find_best_params_classification(name, model, X, Y, test_size_range, random_state_range, max_depth_range):
    best_score = float('-inf')
    best_params = {}

    for test_size, random_state, max_depth in product(test_size_range, random_state_range, max_depth_range):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

        model_instance = model

        if hasattr(model_instance, 'random_state'):
            model_instance.set_params(random_state=random_state)
        if hasattr(model_instance, 'max_depth'):
            model_instance.set_params(max_depth=max_depth)

        model_instance.fit(X_train, Y_train)
        Y_pred = model_instance.predict(X_test)
        score = accuracy_score(Y_test, Y_pred)

        if score > best_score:
            best_score = score
            best_params = {'test_size': test_size, 'random_state': random_state, 'max_depth': max_depth}

    return {'name': name, 'best_params': best_params, 'best_score': best_score}

def find_best_params_regression(name, model, X, Y, test_size_range, random_state_range):
    best_score = float('inf')
    best_params = {}

    for test_size, random_state in product(test_size_range, random_state_range):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

        model_instance = model

        if hasattr(model_instance, 'random_state'):
            model_instance.set_params(random_state=random_state)

        model_instance.fit(X_train, Y_train)
        Y_pred = model_instance.predict(X_test)
        score = mean_squared_error(Y_test, Y_pred)

        if score < best_score:
            best_score = score
            best_params = {'test_size': test_size, 'random_state': random_state}

    return {'name': name, 'best_params': best_params, 'best_score': best_score}

def automl(df):
    try:
        if df is None:
            st.error('Please upload a dataset. If your dataset is prepared for training and testing, go to the next step.', icon="🚨")
            return      
        with st.container(border=True):
            st.info("İşleminiz bir süre devam edecek. Haydi başlayalım!", icon='🎉')
            df, label = data_preprocessing(df)

            # Model seçenekleri ve kısaltmaları
            classification_models = [
                ("LR", LogisticRegression()),
                ("LDA", LinearDiscriminantAnalysis()),
                ("KNN", KNeighborsClassifier()),
                ("DT", DecisionTreeClassifier()),
                ("NB", GaussianNB()),
                ("SVM", SVC()),
                ("RF", RandomForestClassifier()),
                ("GB", GradientBoostingClassifier()),
                ("XGB", XGBClassifier()),
                ("LGBM", LGBMClassifier()),
                ("CatBoost", CatBoostClassifier()),
                ("MLP", MLPClassifier()),
                ("AdaBoost", AdaBoostClassifier()),
                ("Bagging", BaggingClassifier()),
                ("ExtraTrees", ExtraTreesClassifier()),
                ("GaussianProcess", GaussianProcessClassifier())
            ]
            regression_models = [
                ("LinearRegression", LinearRegression()),
                ("Ridge", Ridge()),
                ("Lasso", Lasso()),
                ("ElasticNet", ElasticNet())
            ]

            # Hedef değişkeni seçimi ve problem türü seçimi
            problem_type = st.selectbox("Problemin türünü seçin:", options=['Sınıflandırma', 'Regresyon'])
            hedef_degisken = st.selectbox("Hedef Değişkeni Seçin", df.columns.tolist())
            #st.write(problem_type)

            if hedef_degisken in df.columns:
                X = df.drop(hedef_degisken, axis=1)  # Bağımsız değişkenler
                Y = df[hedef_degisken]  # Hedef değişken
            else:
                st.write("Hedef değişken adı geçersiz. Lütfen mevcut bir hedef değişken adı girin.")
                return
            
            if problem_type == "Sınıflandırma":
                all_models = [name for name, _ in classification_models]
                first_model = st.selectbox("İlk modeli seçin:", all_models, index=0)
                second_model = st.selectbox("İkinci modeli seçin:", all_models, index=0)
                selected_models = [first_model, second_model]
                
            else:
                selected_models = ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']

            ml_start = st.button("Makine Öğrenmesini Başlat.")
            
            if ml_start:
                st.info("Makine öğrenmesi başlatıldı. Lütfen sonuçları bekleyiniz.", icon='🤖')

                if problem_type == 'Sınıflandırma':
                    st.info("Sınıflandırma yapılıyor.", icon="ℹ️")
                    models = [model for name, model in classification_models if name in selected_models]
                    find_best_params = find_best_params_classification
                    st.info("Sınıflandırma best parametreleri bulundu.", icon="ℹ️")
                    
                    test_size_range = [0.1, 0.2, 0.3, 0.4]
                    random_state_range = [42, 2021, 12345]
                    max_depth_range = [3, 5, 7, 9]
                
                elif problem_type == 'Regresyon':
                    st.info("Regresyon yapılıyor.", icon="ℹ️")
                    models = [model for name, model in regression_models if name in selected_models]
                    find_best_params = find_best_params_regression
                    st.info("Regresyon best parametreleri bulundu. Bekleyiniz.", icon="ℹ️")
                    
                    test_size_range = [0.1, 0.2, 0.3, 0.4]
                    random_state_range = [42, 2021, 12345]
                    max_depth_range = []
                
                futures = []
                if problem_type == 'Sınıflandırma':
                    with ProcessPoolExecutor() as executor:
                        for name, alg in classification_models:
                            if name in selected_models:
                                futures.append(
                                    executor.submit(find_best_params, name, alg, X, Y, test_size_range, random_state_range, max_depth_range))
                elif problem_type == 'Regresyon':
                    with ProcessPoolExecutor() as executor:
                        for name, alg in regression_models:
                            if name in selected_models:
                                futures.append(
                                    executor.submit(find_best_params, name, alg, X, Y, test_size_range, random_state_range))
                
                results = []
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)

                df_models = pd.DataFrame({
                    "Models": selected_models,
                    })
                
                # Sonuçları DataFrame'e dönüştürme
                df_results = pd.DataFrame(results)

                #    Streamlit arayüzünde tabloyu gösterme
                st.write("Seçilen Modeller")
                st.write(df_models)
                st.write("Sonuçlar")
                st.write(df_results)
       
    except Exception as e:
        st.write(f"Hata: {e}")

#MANUALML FONKSİYONLARI
def save_model(model, format_choice, filename):
    if format_choice == 'joblib':
        joblib.dump(model, filename)
        st.write(f"Model başarıyla '{filename}' adlı dosyaya kaydedildi.")
    elif format_choice == 'pickle':
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        st.write(f"Model başarıyla '{filename}' adlı dosyaya kaydedildi.")
    elif format_choice == 'onnx':
        st.write("Bu model ONNX formatında kaydedilemez.")
    else:
        st.write("Geçersiz format seçimi! Lütfen 'joblib', 'pickle', 'h5', 'onnx', 'json' veya 'yaml' şeklinde bir format seçin.")
def get_input_value(key):
    if key not in st.session_state:
        st.session_state[key] = ""
    return st.session_state[key]


def predict_new_data(df, model, columns, label_encoder):
    new_data = {}
    row_values = df.iloc[5]
    st.write(row_values)
    
    for column, value in zip(df.columns, row_values):
        new_data[column] = value

    print(new_data)
    
    new_df = pd.DataFrame(new_data, index=[0])

    # Yeni verilerin dönüştürülmesi
    string_columns = new_df.select_dtypes(include=['object']).columns
    new_df[string_columns] = new_df[string_columns].apply(label_encoder.fit_transform)
    new_df = new_df.reindex(columns=columns, fill_value=0)

    st.write("Oluşturulan Sözlük:")
    st.write(new_data)

    prediction = model.predict(new_df)
    st.write("Tahmin edilen hedef değişken:", prediction)

# Kullanılabilir modeller ve bunların adları
models = [
        ("LR", LogisticRegression),
        ("LIR", LinearRegression),
        ("LDA", LinearDiscriminantAnalysis),
        ("KNN", KNeighborsClassifier),
        ("DT", DecisionTreeClassifier),
        ("NB", GaussianNB),
        ("SVM", SVC),
        ("RF", RandomForestClassifier),
        ("GB", GradientBoostingClassifier),
        ("XGB", XGBClassifier),
        ("LGBM", LGBMClassifier),
        ("CatBoost", CatBoostClassifier),
        ("MLP", MLPClassifier),
        ("AdaBoost", AdaBoostClassifier),
        ("Bagging", BaggingClassifier),
        ("ExtraTrees", ExtraTreesClassifier),
        ("GaussianProcess", GaussianProcessClassifier),
        ("Ridge Regression", Ridge),
        ("Lasso Regression", Lasso),
        ("ElasticNet Regresyon", ElasticNet)
    ]


def manualml(df):
    if df is None:
        st.error('Please upload a dataset. If your dataset is prepared for training and testing, go to the next step.', icon="🚨")
        return
    
    st.info("İşleminiz bir süre devam edecek. Haydi başlayalım!", icon='🎉')
     # data_preprocessing fonksiyonu tanımlanmalı veya bu satır kaldırılmalı
    # Veri setini uygun hale getirme
    df.drop_duplicates(inplace=True)
    df.columns = [kolon.lower() for kolon in df.columns]

    # Label Encoding işlemi
    label_encoder = LabelEncoder()
    string_columns = df.select_dtypes(include=['object']).columns
    df[string_columns] = df[string_columns].apply(label_encoder.fit_transform)
    st.write("Label Encoding işlemi tamamlandı.")

    # Eksik değerleri doldurma
    integer_columns = df.select_dtypes(include=['int', 'float']).columns
    df[integer_columns] = df[integer_columns].fillna(df[integer_columns].mean())

    string_columns = df.select_dtypes(include=['object']).columns
    df[string_columns] = df[string_columns].fillna("Bilinmiyor")
    st.write("Eksik değerler dolduruldu.")


    # X ve y ayrıştırma
    target_variable = st.selectbox("Hedef değişkeni seçin:", df.columns)
    X = df.drop(columns=[target_variable])
    y = df[target_variable]

    # Kullanıcıdan test_size ve random_state değerlerini al
    # Kullanıcıdan model seçmesini iste
    
    test_size = st.number_input("Test setinin oranını girin (örn: 0.2):", min_value=0.1, max_value=0.9, step=0.1, value=0.2)
    random_state = st.number_input("Random state değerini girin (örn: 42):", min_value=30, step=2, value=30)

    all_models = [name for name, _ in models]
    selected_model_name = st.selectbox("Model Seçin: ", all_models)
    selected_model_class = [model for name, model in models if name in selected_model_name]

    
    if selected_model_name in ['LR', 'LIR','SVM', 'MLP', 'GaussianProcess', 'Ridge Regression,', 'Lasso Regression', 'ElasticNet Regresyon']:
        pass
    elif selected_model_name in ["LDA", "KNNİ" "NB", "CatBoost"]:
        pass
    elif selected_model_name == 'DT':
        max_depth = st.number_input("max_depth oranını girin (örn: 0.2):", min_value=1, max_value=50, step=2, value=1)

    elif selected_model_name in ['RF', 'GB', 'XGB', 'LGBM', 'ExtraTrees']:
        max_depth = st.number_input("max_depth oranını girin (örn: 0.2):", min_value=1, max_value=50, step=2, value=1)
        n_estimators = st.number_input("n_estimators oranını girin (örn: 0.2):", min_value=10, max_value=100, step=5, value=10)

    elif selected_model_name in ['Bagging', 'AdaBoost']:
        n_estimators = st.number_input("n_estimators oranını girin (örn: 0.2):", min_value=10, max_value=100, step=5, value=10)
    else: 
        pass

    trainModel = st.button("Modeli Eğit.")

    if trainModel:
        st.info('Makine öğrenmesi başlamıştır, Lütfen bekleyiniz.', icon="ℹ️")

        selected_model_class = selected_model_class[0]

        # Seçilen modelin parametrelerini kullanıcıya göster
        params = {}
        st.write(f"Seçilen model: {selected_model_name}")
        st.write("Bu modelin alabileceği parametreler:")

        # Model parametrelerini kullanıcıdan al
        if 'n_estimators' in selected_model_class().get_params():
            params['n_estimators'] = n_estimators
        if 'random_state' in selected_model_class().get_params():
            params['random_state'] = random_state
        if 'max_depth' in selected_model_class().get_params():
            params['max_depth'] = max_depth

        st.write(params)

        # Modeli parametrelerle oluştur
        model = selected_model_class(**params)
        st.write(model)

        # Veriyi eğitim ve test setine ayır
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Modeli eğit ve test et
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Modelin parametrelerini yazdırma
        # st.write("Model Parametreleri:", model.get_params())

        # Sonuçları değerlendir
        if selected_model_name in ["LIR", "Ridge Regression", "Lasso Regression", "ElasticNet Regresyon"]:
            # Regresyon modelleri için
            mse = mean_squared_error(y_test, y_pred)
            st.write("Mean Squared Error:", mse)
        else:
            # Sınıflandırma modelleri için
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Sonuçları yazdırma
            st.write('*Sonuçlar*')
            st.write("Accuracy:", accuracy)
            st.write("Precision:", precision)
            st.write("Recall:", recall)
            st.write("F1 Score:", f1)

            # Yeni veri ile tahmin yapma
        predict_new_data(df, model, X.columns, label_encoder)
            
        #filename = input("Kaydetmek istediğiniz dosyanın adını girin (örn: model.pkl, model.h5, model.onnx, model.json, model.yaml): ")
        #save_model(model, format_choice, filename)   

# FRONTEND FONKSİYONU
def frontend():
    st.subheader("Machine Learning")
    st.write("Makine öğrenmesi, bilgisayarların açıkça programlanmadan verilerden öğrenmesini ve tahminler yapmasını sağlayan bir yapay zeka alt alanıdır. Bu süreçte, bilgisayarlar verilere dayalı olarak kalıpları ve ilişkileri öğrenir ve bu öğrenme sonucunda gelecekteki verilere uygulanan tahminlerde bulunurlar. Makine öğrenmesi, çeşitli algoritmalar ve teknikler kullanarak verilerden anlamlı sonuçlar çıkarır. Makine öğrenmesinin temel bileşenleri veri, model ve algoritmadır. Veri, modelin öğrenmesi için kullanılan örnekleri içerir. Model, verilerden öğrenilen matematiksel bir temsildir. Algoritma ise modeli eğitmek için kullanılan yöntemdir. Makine öğrenmesi, çok çeşitli alanlarda kullanılır. Tahmin ve sınıflandırma en yaygın kullanım alanları arasındadır. Makine öğrenmesi, büyük miktarda veriyi analiz etmek ve bu verilere dayalı kararlar almak için güçlü bir araçtır. Bu, iş dünyasından sağlık sektörüne kadar birçok alanda büyük avantajlar sağlar. Makine öğrenmesi sayesinde, daha doğru tahminler yapılabilir, verimlilik artırılabilir ve yeni keşifler yapılabilir.")
    with st.expander("🔗 Classification"):
        st.write("Veri işleme ve model seçimi işlemlerini otomatikleştirmek için bu aracı kullanabilirsiniz.")
    with st.expander("🔗 Regression"):
        st.write("Veri işleme ve model seçimi işlemlerini otomatikleştirmek için bu aracı kullanabilirsiniz.")
    
    
    st.info('If you want to know more in detail, read our [**manual**](https://www.retmon.com/blog/veri-gorsellestirme-nedir#:~:text=Veri%20g%C3%B6rselle%C5%9Ftirme%3B%20verileri%20insan%20beyninin,i%C3%A7in%20kullan%C4%B1lan%20teknikleri%20ifade%20eder.).', icon="ℹ️")
   
    # Dosya yükleme işlemi için Streamlit'in file_uploader fonksiyonunu kullanma
    uploaded_file = st.file_uploader("Upload a file", type=["csv", "yaml", "json", "zip"])

    ml_option = st.selectbox("Do you want to do your training yourself or should it be done automatically?", ["Choose", "I'll do it myself.", "Do it."])

    if uploaded_file is None:
        st.error('Please upload a dataset. If your dataset is prepared for training and testing, go to the next step.', icon="🚨")
    elif ml_option == "I'll do it myself.":
        df = handle_file_upload(uploaded_file)
        manualml(df)
    elif ml_option == "Do it.":
        df = handle_file_upload(uploaded_file)
        automl(df)
    else:
        st.write("Please made an valid selection.")

if __name__ == "__main__":
    # Page configuration
    st.set_page_config(page_title="ATOM AI", layout="wide",)
    # Yakınsama uyarılarını bastırmak için ayarları yapılandırma
    filterwarnings("ignore", category=ConvergenceWarning)

    #Frontend ile arayüzün oluşturulması.
    frontend()
