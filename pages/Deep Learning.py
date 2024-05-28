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

def save_model():
    pass

def modeli_olustur_ve_egit(data_dir, num_classes):
    use_datagen = st.selectbox("ImageDataGenerator kullanmak ister misiniz? (evet/hayır): ", ["evet", "hayır"])

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

    optimizers = {
        'adam': Adam,
        'rmsprop': RMSprop,
        'sgd': SGD,
        'adagrad': Adagrad,
        'adadelta': Adadelta,
        'adamax': Adamax,
        'nadam': Nadam,
        'ftrl': Ftrl
    }

    activation_options = ['relu', 'sigmoid', 'tanh', 'softmax']
    loss_options = ['categorical_crossentropy', 'binary_crossentropy', 'mean_squared_error']
    metrics_options = ['accuracy', 'precision', 'recall', 'mean_absolute_error']

    feature_extractors = [
        MobileNetV2(include_top=False, input_shape=(224, 224, 3)),
        EfficientNetB0(include_top=False, input_shape=(224, 224, 3)),
        InceptionV3(include_top=False, input_shape=(224, 224, 3)),
        ResNet50(include_top=False, input_shape=(224, 224, 3)),
        DenseNet121(include_top=False, input_shape=(224, 224, 3)),
        VGG16(include_top=False, input_shape=(224, 224, 3)),
        Xception(include_top=False, input_shape=(224, 224, 3)),
        NASNetMobile(include_top=False, input_shape=(224, 224, 3))
    ]
    feature_extractor_names = ['mobilenetv2', 'efficientnetb0', 'inceptionv3', 'resnet50', 'densenet121', 'vgg16', 'xception', 'nasnetmobile']

    def get_user_choice(prompt, options, default):
        choice = st.text_input(f"{prompt} {options}: ").lower()
        if choice not in options:
            st.write(f"Geçersiz seçenek. '{default}' kullanılacak.")
            return default
        return choice

    optimizer_name = get_user_choice("Kullanılacak optimizasyon algoritmasını seçin", optimizers.keys(), 'adam')
    activation = get_user_choice("Kullanılacak aktivasyon fonksiyonunu seçin", activation_options, 'relu')
    loss = get_user_choice("Kullanılacak loss fonksiyonunu seçin", loss_options, 'categorical_crossentropy')
    metrics = get_user_choice("Kullanılacak metrics fonksiyonunu seçin", metrics_options, 'accuracy')
    feature_extractor_choice = get_user_choice("Kullanılacak özellik çıkarıcıyı seçin ya da en iyi ile en uygun özellik çıkarıcıyı bulup kullansın", feature_extractor_names + ['en iyi'], 'mobilenetv2')

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

    if feature_extractor_choice == 'en iyi':
        def train_and_evaluate(feature_extractor, optimizer, train_generator, valid_generator):
            model = Sequential()
            model.add(feature_extractor)
            model.add(GlobalAveragePooling2D())
            model.add(Dense(num_classes, activation=activation))

            model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

            model.fit(
                train_generator,
                steps_per_epoch=len(train_generator),
                epochs=3,  # 3 epoch for quick evaluation
                validation_data=valid_generator,
                validation_steps=len(valid_generator),
                verbose=1
            )

            _, accuracy = model.evaluate(valid_generator, steps=len(valid_generator), verbose=1)
            return model, accuracy

        best_accuracy = 0
        best_model = None
        best_feature_extractor = None

        for feature_extractor, feature_extractor_name in zip(feature_extractors, feature_extractor_names):
            optimizer = Adam()  # Varsayılan optimizer
            st.write(f"Training with {feature_extractor_name}...")


            model, accuracy = train_and_evaluate(feature_extractor, optimizer, train_generator, valid_generator)
            st.write(f"Validation accuracy: {accuracy}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_feature_extractor = feature_extractor

        st.write(f"Best feature extractor: {best_feature_extractor}")
        st.write(f"Best validation accuracy: {best_accuracy}")

        feature_extractor = best_feature_extractor

    else:
        feature_extractor = feature_extractors[feature_extractor_names.index(feature_extractor_choice)]

    optimizer = optimizers[optimizer_name]()

    model = Sequential()
    model.add(feature_extractor)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1024, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
    callbacks = [early_stopping, model_checkpoint]

    use_lr_scheduler = input("LearningRateScheduler kullanmak ister misiniz? (evet/hayır): ").lower()
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
    print(f"Final validation accuracy: {accuracy}")



    return model, {
        'optimizer': optimizer_name,
        'activation': activation,
        'loss': loss,
        'metrics': metrics,
        'feature_extractor': feature_extractor_choice
    }

def tahmin_yap_tabular(model, data_type, num_classes, target_variable, dataset_path):
    if data_type == 'Tabular':
        df = pd.read_csv(dataset_path)

        df.columns = [kolon.lower() for kolon in df.columns]
        feature_names = df.drop(columns=[target_variable]).columns.tolist()

        print("Özellik isimleri:", feature_names)
        features = []
        for feature_name in feature_names:
            feature_value = input(f"{feature_name} değerini girin: ")
            features.append(feature_value)

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
    if data_type == 'Image':
        from keras.preprocessing import image
        #image_path = input("Tahmin edilecek görüntünün yolunu girin: ")
        image_path = st.file_uploader("Upload a file", type=["csv", "yaml", "json", "zip"])
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
        st.write(f"Tahmin edilen sınıf: {predicted_class_name}")

    else:
        print("Geçersiz veri tipi. Tahmin yapılamıyor.")

def main():
    data_type = st.selectbox("Veri tipi seçin:", ["Tabular", "Image"])
    if data_type == 'Tabular':
        uploaded_file = st.file_uploader("Upload a file", type=["csv", "yaml", "json", "zip"])
        if uploaded_file is None:
            st.error('Please upload a dataset. If your dataset is prepared for training and testing, go to the next step.', icon="🚨")
        else:
            df =  handle_file_upload(uploaded_file)
            
        target_variable = st.selectbox("Hedef değişkenin adını girin:", df.columns.to_list())

        X, Y, num_classes= preprocess_tabular_data(df, target_variable)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        input_shape = X_train.shape[1]

        # Parametreler
        manual_params = st.selectbox("Model parametrelerini manuel girmek ister misiniz? (evet/hayır):", ["Evet", "Hayır"])
        if manual_params == "Evet":
            optimizers = {
                'adam': Adam,
                'rmsprop': RMSprop,
                'sgd': SGD,
                'adagrad': Adagrad,
                'adadelta': Adadelta,
                'adamax': Adamax,
                'nadam': Nadam,
                'ftrl': Ftrl
            }
            activation_options = ["adam", 'relu', 'sigmoid', 'tanh', 'softmax']
            loss_options = ['categorical_crossentropy', 'binary_crossentropy', 'mean_squared_error']
            metrics_options = ['accuracy', 'precision', 'recall', 'mean_absolute_error']

            optimizer = st.selectbox("Kullanılacak optimizasyon algoritmasını seçin", optimizers.keys())
            activation = st.selectbox("Kullanılacak aktivasyon fonksiyonunu seçin", activation_options)
            loss = st.selectbox("Kullanılacak loss fonksiyonunu seçin", loss_options)
            metrics = st.selectbox("Kullanılacak metrics fonksiyonunu seçin", metrics_options)

            model = build_tabular_model(X_train.shape[1], num_classes, optimizer=optimizer, activation=activation,
                                        loss=loss, metrics=metrics)
            model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.2)
            score = model.evaluate(X_test, Y_test, verbose=0)
            st.write(f"Doğruluk skoru: {score}")

        else:
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

            model = build_tabular_model(**best_params)
            model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.2)
            st.write(f"En iyi parametreler: {best_params}")
            st.write(f"En iyi doğruluk: {best_score}")

        if st.selectbox("Modeli kaydetmek ister misiniz? (evet/hayır): ", ["Evet", "Hayır"]) == 'Evet':
            format_choice = st.selectbox("Kaydetme formatını seçin (joblib, pickle, h5):", ['joblib', 'pickle', 'h5'])
            filename = st.text_input("Modeli kaydetmek için dosya adını girin (örn: 'model.joblib'): ")
            save_model(model, format_choice, filename)
        if st.selectbox("Tahmin yapmak ister misiniz? (evet/hayır): ", ["Evet", "Hayır"]) == 'Evet':
            tahmin_yap_tabular(model, data_type, num_classes, target_variable, uploaded_file)

    elif data_type == 'Image':
        #data_dir = st.text_input("Görüntü verilerinin bulunduğu dizini girin (örn: 'data/'): ")
        st.caption('Görüntü verilerinin bulunduğu dizini girin')
        data_dir = st.file_uploader("Upload a file", type=["csv", "yaml", "json", "zip"])
        # Dosya sayısını bularak num_classes belirleme
        train_dir = os.path.join(data_dir, 'train')
        num_classes = len(os.listdir(train_dir))
        st.write(f"Belirlenen sınıf sayısı: {num_classes}")
        model, best_params = modeli_olustur_ve_egit(data_dir, num_classes)

        st.write("Kullanılan parametreler:")
        for param, value in best_params.items():
            st.write(f"{param}: {value}")

        if st.selectbox("Modeli kaydetmek ister misiniz? (evet/hayır): ", ["Evet", "Hayır"]) == 'Evet':
            format_choice = st.selectbox("Kaydetme formatını seçin (joblib, pickle, h5): ", ["joblib", "pickle", "h5"])
            filename = st.text_input("Modeli kaydetmek için dosya adını girin (örn: 'model.joblib'): ")
            save_model(model, format_choice, filename)
        if st.selectbox("Tahmin yapmak ister misiniz? (evet/hayır): ", ["Evet", "Hayır"]) == 'Evet':
            (tahmin_yap_image(model, data_type, num_classes, data_dir))

    else:
        st.write("Geçersiz veri tipi. Lütfen 'tabular' veya 'image' girin.")



if __name__ == "__main__":
    main()