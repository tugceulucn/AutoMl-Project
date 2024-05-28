import streamlit as st
import os
import keras, yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.applications import MobileNetV2, EfficientNetB0, InceptionV3, ResNet50, DenseNet121, VGG16, Xception, \
    NASNetMobile
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop, SGD, Adagrad, Adadelta, Adamax, Nadam, Ftrl
from keras.callbacks import LearningRateScheduler
from keras.utils import to_categorical
import joblib
import pickle, zipfile
import json
import itertools
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="ATOM AI",
    layout="wide",
)

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

def build_tabular_model(input_shape, num_classes, optimizer='adam', activation='relu', loss='categorical_crossentropy',
                        metrics='accuracy'):
    model = Sequential()
    model.add(Dense(64, input_shape=(input_shape,), activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation=activation))
    model.add(Dense(num_classes, activation='softmax'))  # categorical classification için
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
    return model

# En iyi parametreleri bulma fonksiyonu
def grid_search_for_best_params(build_fn, X_train, Y_train, X_val, Y_val, param_grid):
    best_score = -np.inf
    best_params = None

    from itertools import product
    for params in product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        model = build_fn(**param_dict)
        model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=0)
        score = model.evaluate(X_val, Y_val, verbose=0)
        if score[1] > best_score:  # Assuming that the metric is accuracy at index 1
            best_score = score[1]
            best_params = param_dict

    return best_params, best_score

def preprocess_tabular_data(df, target_variable):
    df.drop_duplicates(inplace=True)
    df.columns = [kolon.lower() for kolon in df.columns]
    X = df.drop(columns=[target_variable])
    Y = df[target_variable].values.reshape(-1, 1)
    # One-Hot Encoding for the target variable
    encoder = OneHotEncoder(sparse_output=False)
    Y = encoder.fit_transform(Y)
    # One-Hot Encoding for categorical features
    categorical_features = X.select_dtypes(include=['object']).columns
    numeric_features = X.select_dtypes(include=['number']).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(sparse_output=False), categorical_features)
        ])

    X = preprocessor.fit_transform(X)

    num_classes = Y.shape[1]  # Automatically determine the number of classes

    return X, Y, num_classes

def save_model(model, format_choice, filename):
    if format_choice == 'joblib':
        joblib.dump(model, filename)
        st.write(f"Model başarıyla '{filename}' adlı dosyaya kaydedildi.")
    elif format_choice == 'pickle':
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        st.write(f"Model başarıyla '{filename}' adlı dosyaya kaydedildi.")
    elif format_choice == 'h5':
        if isinstance(model, Sequential):
            model.save(filename)
            st.write(f"Model başarıyla '{filename}' adlı dosyaya HDF5 formatında kaydedildi.")
        else:
            st.write("Bu model HDF5 formatında kaydedilemez.")
    else:
        st.write("Geçersiz format seçimi! Lütfen 'joblib', 'pickle' veya 'h5'  şeklinde bir format seçin.")

def tahmin_yap_tabular(model, data_type, num_classes, target_variable, df):
    if data_type == 'Tabular':
        target_variable = target_variable.lower()
        df.columns = [kolon.lower() for kolon in df.columns]
        feature_names = df.drop(columns=[target_variable]).columns.tolist()

        st.write("Özellik isimleri:", feature_names)
        
        features = []
        feature_value = st.text_input(f"{feature_names} değerlerini girin: ")
        feature_value = feature_value.split()
        
    
        # Her bir öğeyi uygun türde bir değere dönüştür
        for i in feature_value:
                    features.append(float(i))

        st.write(features)

        if st.button("okey"):
            # Create a DataFrame for the input features to ensure proper preprocessing
            input_data = pd.DataFrame([features], columns=feature_names)

            # Load the preprocessor
            with open('preprocessor.pkl', 'rb') as f:
                preprocessor = pickle.load(f)

            # Preprocess the input data
            processed_features = preprocessor.transform(input_data)

            # Ensure the shape is compatible with the model
            processed_features = processed_features.reshape(1, -1)

            prediction = model.predict(processed_features)
            predicted_class = np.argmax(prediction, axis=1)
            st.write(f"Tahmin edilen sınıf: {predicted_class[0]}")

def deepLearning():
    data_type = st.selectbox("Veri tipi seçin:", ["Tabular", "Image"])

    if data_type == 'Tabular':
        uploaded_file = st.file_uploader("Upload a file", type=["csv", "yaml", "json", "zip"])
        if uploaded_file is None:
            st.error('Please upload a dataset. If your dataset is prepared for training and testing, go to the next step.', icon="🚨")
        else:
            df =  handle_file_upload(uploaded_file)

            target_variable = st.selectbox("Hedef değişkenin adını girin:", df.columns.to_list())
            X, Y, num_classes= preprocess_tabular_data(df, target_variable.lower())
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            input_shape = X_train.shape[1]
            # Parametreler
            manual_params = st.selectbox("Model parametrelerini manuel girmek ister misiniz?", ["Evet", "Hayır"])
            if manual_params == "Evet":
                optimizers = {'adam': Adam, 'rmsprop': RMSprop, 'sgd': SGD, 'adagrad': Adagrad, 'adadelta': Adadelta, 'adamax': Adamax, 'nadam': Nadam,'ftrl': Ftrl}
                activation_options = ['relu', 'sigmoid', 'tanh', 'softmax']
                loss_options = ['categorical_crossentropy', 'binary_crossentropy', 'mean_squared_error']
                metrics_options = ['accuracy', 'precision', 'recall', 'mean_absolute_error']
            
                def get_user_choice(prompt, options, default):
                    choice = st.selectbox(f"{prompt} {options}: ").lower()
                    if choice not in options:
                        st.write(f"Geçersiz seçenek. '{default}' kullanılacak.")
                        return default
                    return choice

                col1, col2, col3, col4 = st.columns(4)
                with col1:    
                    optimizer = st.selectbox("Kullanılacak optimizasyon algoritmasını seçin", optimizers.keys())
                with col2:    
                    activation = st.selectbox("Kullanılacak aktivasyon fonksiyonunu seçin", activation_options)
                with col3:    
                    loss = st.selectbox("Kullanılacak loss fonksiyonunu seçin", loss_options)
                with col4:
                    metrics = st.selectbox("Kullanılacak metrics fonksiyonunu seçin", metrics_options)
                
                if st.button("Parametreler tamamlandı."):
                    model = build_tabular_model(X_train.shape[1], num_classes, optimizer=optimizer, activation=activation, loss=loss, metrics=metrics)
                    model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.2)
                    score = model.evaluate(X_test, Y_test, verbose=0)
                    st.write(f"Doğruluk skoru: {score}")

                if st.selectbox("Modeli kaydetmek ister misiniz?", ["Evet", "Hayır"]) == 'Evet':
                        format_choice = st.selectbox("Kaydetme formatını seçin (joblib, pickle, h5):", ['joblib', 'pickle', 'h5'])
                        filename = st.text_input("Modeli kaydetmek için dosya adını girin (örn: 'model.joblib'): ")
                        #save_model(model, format_choice, filename)

                if st.selectbox("Tahmin yapmak ister misiniz?", ["Evet", "Hayır"]) == 'Evet':
                         tahmin_yap_tabular(model, data_type, num_classes, target_variable, df)
            

            else:
                if st.button("Parametreler tamamlandı."):
                    param_grid = {
                        'input_shape': [X_train.shape[1]],
                        'num_classes': [num_classes],
                        'optimizer': ['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta'],
                        'activation': ['relu', 'tanh', 'sigmoid', 'softmax'],
                        'loss': ['categorical_crossentropy', 'mean_squared_error', 'binary_crossentropy'],
                        'metrics': ['accuracy']
                    }

                    best_params, best_score = grid_search_for_best_params(build_tabular_model, X_train, Y_train, X_test, Y_test,
                                                                        param_grid)
                    if st.button("Parametreler tamamlandı."):
                        model = build_tabular_model(**best_params)
                        model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.2)
                        st.write(f"En iyi parametreler: {best_params}")
                        st.write(f"En iyi doğruluk: {best_score}")

                    if st.selectbox("Modeli kaydetmek ister misiniz?", ["Evet", "Hayır"]) == 'Evet':
                        format_choice = st.selectbox("Kaydetme formatını seçin (joblib, pickle, h5):", ['joblib', 'pickle', 'h5'])
                        filename = st.text_input("Modeli kaydetmek için dosya adını girin (örn: 'model.joblib'): ")
                        #save_model(model, format_choice, filename)

                    if st.selectbox("Tahmin yapmak ister misiniz?", ["Evet", "Hayır"]) == 'Evet':
                         tahmin_yap_tabular(model, data_type, num_classes, target_variable, df)

            

            
    
    elif data_type == 'Image':
        uploaded_file = st.file_uploader("Upload a file", type=["csv", "yaml", "json", "zip"])
        if uploaded_file is None:
            st.error('Please upload a dataset. If your dataset is prepared for training and testing, go to the next step.', icon="🚨")
        else:
            df =  handle_file_upload(uploaded_file)

            # Dosya sayısını bularak num_classes belirleme
            train_dir = os.path.join(df, 'train')
            num_classes = len(os.listdir(train_dir))
            st.write(f"Belirlenen sınıf sayısı: {num_classes}")

    else:
        st.write("Lütfen geçerli bir seçenek seçiniz.")

if __name__ == "__main__":
    deepLearning()