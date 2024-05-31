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
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, RMSprop, SGD, Adagrad, Adadelta, Adamax, Nadam, Ftrl
from keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import pickle, zipfile
import json
import itertools
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.applications import MobileNetV2, EfficientNetB0, InceptionV3, ResNet50, DenseNet121, VGG16, Xception, \
    NASNetMobile
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop, SGD, Adagrad, Adadelta, Adamax, Nadam, Ftrl
from keras.callbacks import LearningRateScheduler
from keras.utils import to_categorical
import joblib
import pickle
import json
import itertools
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
# Page configuration
st.set_page_config(
    page_title="ATOM AI",
    layout="wide",
)

# 1. File Upload Function
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
                df = pd.DataFrame(yaml_verisi)
                df.to_csv("veri.csv", index=False)
                df = pd.read_csv("veri.csv")
                st.success("YAML Data loaded successfully. Let's continue with Machine Learning!", icon="✅")
            elif uploaded_file.type == 'application/json':  # JSON dosyası yüklenirse
                json_verisi = json.load(uploaded_file)
                df = pd.DataFrame(json_verisi)
                df.to_csv("veri.csv", index=False)
                df = pd.read_csv("veri.csv")
                st.success("JSON Data loaded successfully. Let's continue with Machine Learning!", icon="✅")
            elif uploaded_file.type in ['application/zip', 'application/x-zip-compressed']:  # Zip dosyası yüklenirse (resim klasörü)
                with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                    zip_ref.extractall("extracted_images")
                    data_dir="extracted_images/train"
                    return data_dir
                st.success("ZIP/Image folder has been successfully uploaded and opened. Let's continue with Machine Learning!", icon="✅")
            else:
                st.write("The file format is not supported.")
                
            if df.empty:
                st.error('The loaded data set is empty.', icon="🚨")
            
            return df
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        st.error("No columns to parse from file. Please check the file content.", icon="🚨")
    except Exception as e:
        st.error(f"Error reading the file: {e}", icon="🚨")

# 2. Tabular Model Building Function
def build_tabular_model(input_shape, num_classes, optimizer='adam', activation='relu', loss='categorical_crossentropy', metrics='accuracy'):
    model = Sequential()
    model.add(Dense(64, input_shape=(input_shape,), activation=activation, name="dense_2"))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation=activation, name="dense_3"))
    model.add(Dense(num_classes, activation='softmax', name="dense_1"))  # categorical classification için
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
    return model

# 3. Grid Search for Best Parameters Function
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

# 4. Data Preprocessing Function
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

# 5. Model Saving Function
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

# 6. Tabular Data Prediction Function
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

        if st.button("okey", key="unique_key_okey"):
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
def tahmin_yap_image(model, data_type, num_classes, data_dir):
    if data_type == 'image':
        from keras.preprocessing import image
        image_path = input("Tahmin edilecek görüntünün yolunu girin: ")
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)
        predicted_class_idx = np.argmax(prediction, axis=1)[0]
        datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = datagen.flow_from_directory(
            os.path.join(data_dir, 'train'),
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=True
        )
        # Save class indices
        class_indices = train_generator.class_indices
        class_names = {v: k for k, v in class_indices.items()}

        predicted_class_name = class_names[predicted_class_idx]
        print(f"Tahmin edilen sınıf: {predicted_class_name}")

    else:
        print("Geçersiz veri tipi. Tahmin yapılamıyor.")

def modeli_olustur_ve_egit(data_dir, num_classes,model_type,optimizer,loss,metrics,use_datagen,metric,activation):
    if use_datagen == 'evet':
        rescale = 1. / 255
        rotation_range = 20
        width_shift_range = 0.2
        height_shift_range = 0.2
        shear_range = 0.2
        zoom_range = 0.2
        horizontal_flip = True

        st.write("Sabit ImageDataGenerator parametreleri kullanılıyor:")
        st.write(f"Rescale: {rescale}")
        st.write(f"Döndürme Aralığı: {rotation_range}")
        st.write(f"Genişlik Kaydırma Aralığı: {width_shift_range}")
        st.write(f"Yükseklik Kaydırma Aralığı: {height_shift_range}")
        st.write(f"Eğim Aralığı: {shear_range}")
        st.write(f"Zoom Aralığı: {zoom_range}")
        st.write(f"Yatay Çevirme: {horizontal_flip}")

        datagen = ImageDataGenerator(
            rescale=rescale,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip,
            fill_mode='nearest'
        )
    else:
        datagen = ImageDataGenerator(rescale=1. / 255)
       
    model = Sequential()
    model.add(model_type)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1024, activation=activation, name='dense_4'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax', name='dense_5'))

    model.compile(optimizer=optimizer, loss=loss, metrics=metric)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
    callbacks = [early_stopping, model_checkpoint]

    use_lr_scheduler = st.selectbox("LearningRateScheduler kullanmak ister misiniz?", ['choose', 'evet', 'hayır']).lower()
    if use_lr_scheduler == 'evet':
        def scheduler(epoch, lr):
            return lr * 0.1

        lr_scheduler = LearningRateScheduler(scheduler)
        callbacks.append(lr_scheduler)


    #
    checkpoint_callback = ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    #
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )
    train_generator = datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=True
    )

    valid_generator = datagen.flow_from_directory(
        os.path.join(data_dir, 'valid'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=10,
        validation_data=valid_generator,
        validation_steps=len(valid_generator),
        verbose=1,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    _, accuracy = model.evaluate(valid_generator, steps=len(valid_generator), verbose=1)
    st.write(f"Final validation accuracy: {accuracy}")

    return model

# 7. Image Data Prediction Function
def tahmin_yap_image(model, data_type, num_classes, data_dir):
    if data_type == 'image':
        from keras.preprocessing import image
        image_path = input("Tahmin edilecek görüntünün yolunu girin: ")
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)
        predicted_class_idx = np.argmax(prediction, axis=1)[0]
        datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = datagen.flow_from_directory(
            os.path.join(data_dir, 'train'),
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=True
        )
        class_labels = list(train_generator.class_indices.keys())
        predicted_class = class_labels[predicted_class_idx]
        st.write(f"Tahmin edilen sınıf: {predicted_class}")

# Streamlit session state management
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False

if 'data' not in st.session_state:
    st.session_state.data = None

if 'model' not in st.session_state:
    st.session_state.model = None

if 'data_type' not in st.session_state:
    st.session_state.data_type = None

if 'target_variable' not in st.session_state:
    st.session_state.target_variable = None

if 'num_classes' not in st.session_state:
    st.session_state.num_classes = None

# Sidebar - Options and parameters
st.sidebar.title("ATOM AI")
user_name = st.sidebar.text_input("Adınızı girin")
uploaded_file = st.sidebar.file_uploader("Veri dosyasını yükleyin", type=['csv', 'yaml', 'json', 'zip'])

if uploaded_file is not None:
    st.session_state.data = handle_file_upload(uploaded_file)
    st.session_state.file_uploaded = True


if st.session_state.file_uploaded:
    data_type = st.sidebar.radio("Veri Tipini Seçin", ['Tabular', 'image'])
    st.session_state.data_type = data_type
    handle_file_upload(uploaded_file)
    data_dir = 'C:/Users/ACER/Documents/GitHub/AutoMl-Project/extracted_images/Köpek/'

        # Dosya sayısını bularak num_classes belirleme
    train_dir = os.path.join(data_dir, "train")
    num_classes = len(os.listdir(train_dir))
    st.write(f"Belirlenen sınıf sayısı: {num_classes}")

    if data_type == 'Tabular':
        st.write("Yüklü verinin ilk 5 satırı:")
        st.write(st.session_state.data.head())

        target_variable = st.sidebar.text_input("Hedef değişkeni girin:")
        st.session_state.target_variable = target_variable.lower()

        # Makine öğrenmesi modelini seçme
        model_type = st.sidebar.selectbox("Model Türünü Seçin", ["Kullanıcı Tanımlı", "Grid Search"])

        if model_type == "Kullanıcı Tanımlı":
            optimizer = st.sidebar.selectbox("Optimizer", ['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam', 'ftrl'])
            activation = st.sidebar.selectbox("Activation Function", ['relu', 'tanh', 'sigmoid'])
            loss = st.sidebar.selectbox("Loss Function", ['categorical_crossentropy', 'binary_crossentropy'])
            metrics = st.sidebar.selectbox("Metrics", ['accuracy'])
            if st.sidebar.button("Modeli Eğit"):
                if st.session_state.data is not None and target_variable is not None:
                    X, Y, num_classes = preprocess_tabular_data(st.session_state.data, target_variable)
                    st.session_state.num_classes = num_classes
                    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
                    model = build_tabular_model(X_train.shape[1], num_classes, optimizer, activation, loss, metrics)
                    model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_val, Y_val))
                    st.session_state.model = model
                    st.success("Model başarıyla eğitildi!")
                    save_model(model, "h5", "trained_model.h5")
        
        elif model_type == "Grid Search":
            param_grid = {
                'optimizer': ['adam', 'sgd'],
                'activation': ['relu', 'tanh'],
                'loss': ['categorical_crossentropy', 'binary_crossentropy'],
                'metrics': ['accuracy']
            }
            if st.sidebar.button("Grid Search ile En İyi Parametreleri Bul ve Modeli Eğit"):
                if st.session_state.data is not None and target_variable is not None:
                    X, Y, num_classes = preprocess_tabular_data(st.session_state.data, target_variable)
                    st.session_state.num_classes = num_classes
                    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
                    best_params, best_score = grid_search_for_best_params(
                        lambda optimizer, activation, loss, metrics: build_tabular_model(X_train.shape[1], num_classes, optimizer, activation, loss, metrics),
                        X_train, Y_train, X_val, Y_val, param_grid
                    )
                    st.write(f"En iyi parametreler: {best_params}")
                    st.write(f"En iyi doğruluk skoru: {best_score}")

                    model = build_tabular_model(X_train.shape[1], num_classes, **best_params)
                    model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_val, Y_val))
                    st.session_state.model = model
                    st.success("Model başarıyla eğitildi!")
                    save_model(model, "h5", "trained_model.h5")

    elif data_type == 'image':
        st.write("Resim verisi seçildi. Lütfen modeli eğitmek için ilerleyin.")
        model_type = st.sidebar.selectbox("Model Türünü Seçin", ['MobilNetV2', 'EfficientNet', 'VGG16'])
        optimizer = st.sidebar.selectbox("Optimizer", ['adam', 'sgd', 'rmsprop'])
        loss = st.sidebar.selectbox("Loss Function", ['categorical_crossentropy', 'binary_crossentropy'])
        metrics = st.sidebar.selectbox("Metrics", ['accuracy'])
        activation =st.sidebar.selectbox("activation_options",['relu', 'sigmoid', 'tanh', 'softmax'])

        use_datagen = st.selectbox("ImageDataGenerator kullanmak ister misiniz?: ", ['choose', 'evet', 'hayır']).lower()
        if st.sidebar.button("Modeli Eğit"):
            if st.session_state.data is not None:
                model=modeli_olustur_ve_egit(data_dir, num_classes,model_type,optimizer,loss,metrics,use_datagen,metrics,activation)
                st.write(f"Model parametreleri:{model,optimizer,activation,loss,metrics,model_type}")

if st.session_state.model is not None:
    st.header("Model Tahmini")
    if st.session_state.data_type == 'Tabular':
        tahmin_yap_tabular(st.session_state.model, st.session_state.data_type, st.session_state.num_classes, st.session_state.target_variable, st.session_state.data)
    elif st.session_state.data_type == 'image':
        tahmin_yap_image(st.session_state.model, st.session_state.data_type, st.session_state.num_classes, "extracted_images")