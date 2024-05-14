import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_card import card
from streamlit_lottie import st_lottie  # pip install streamlit-lottie
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from bokeh.plotting import figure
import numpy as np
# Gerekli kütüphaneler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, mean_squared_error
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="ATOM AI",
    layout="wide",
)

selected = option_menu(None, ["INPUT", "OUTPUT"], 
    icons=["bi bi-gear-fill", "bi bi-rocket-takeoff"], 
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "14px"}, 
        "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "blue"},
    }
) 

#MAKİNE ÖĞRENMESİ
def auto_ml(df, hedef, modeller):
    try:
        # Kullanıcıya hedef değişkeni sor
        hedef_degisken = hedef
        # Hedef değişkeni varsa işlemleri gerçekleştir
        if hedef_degisken in df.columns:
            # Bağımsız değişkenleri ve hedef değişkeni ayırma
            X = df.drop(hedef_degisken, axis=1)  # Bağımsız değişkenler
            Y = df[hedef_degisken]  # Hedef değişken

        else:
            st.warning('Hedef değişken adı geçersiz. Lütfen mevcut bir hedef değişken adı girin.', icon="⚠️")
            return

        st.success('Makine öğrenmesi adımı seçildi!', icon="✅")

        secilen_modeller = modeller

        # Seçilen modeller için en iyi test boyutu ve rastgele durumunun bulunması
        best_models = []
        with ThreadPoolExecutor() as executor:
            futures = []
            for name in secilen_modeller:
                futures.append(executor.submit(find_best_test_size_and_random_state, name, X, Y))

            for future in as_completed(futures):
                best_model = future.result()
                best_models.append(best_model)

        # Modelleri doğruluk skoruna göre sıralama
        best_models.sort(key=lambda x: x['best_score'], reverse=True)

        # Sıralanmış modelleri yazdırma
        st.write("\nSıralanmış Modeller:")
        for idx, model in enumerate(best_models):
            print(f"{idx + 1}. {model['name']} - En İyi Parametreler: {model['best_params']}, Doğruluk: {model['best_score']} (Test Size: {model['test_size']}, Random State: {model['random_state']})")

        # Modeller için sonuçları yazdırma
        for model_info in best_models:
            model_name = model_info['name']
            best_params = model_info['best_params']
            test_size = model_info['test_size']
            random_state = model_info['random_state']

            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
            model = get_model_instance(model_name, best_params)
            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test)

            # Sonuçları hesapla
            accuracy = accuracy_score(Y_test, Y_pred)
            precision = precision_score(Y_test, Y_pred, average='weighted')
            recall = recall_score(Y_test, Y_pred, average='weighted')
            f1 = f1_score(Y_test, Y_pred, average='weighted')
            roc_auc = roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1])
            conf_matrix = confusion_matrix(Y_test, Y_pred)
            mse = mean_squared_error(Y_test, Y_pred)

            # Sonuçları yazdırma
            st.write(f"\n{model_name} Modeli İçin Sonuçlar (Test Size: {test_size}, Random State: {random_state}):")
            st.write("En İyi Parametreler:", best_params)
            st.write("Doğruluk:", accuracy)
            st.write("Hassasiyet:", precision)
            st.write("Geri Çağırma:", recall)
            st.write("F1 Skoru:", f1)
            st.write("ROC-AUC Skoru:", roc_auc)
            st.write("Karmaşıklık Matrisi:\n", conf_matrix)
            st.write("Ortalama Kare Hata (MSE):", mse)

    except Exception as e:
        #print("Bir hata oluştu:", e)
        st.warning(f'Bir hata oluştu:"{e}', icon="⚠️")



def find_best_test_size_and_random_state(name, X, Y):
    best_accuracy = 0.0
    best_test_size = 0.0
    best_random_state = 0

    for test_size in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for random_state in range(101):
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

            model = get_model_instance(name)
            param_grid = get_parameter_grid(name)

            grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train, Y_train)

            if grid_search.best_score_ > best_accuracy:
                best_accuracy = grid_search.best_score_
                best_test_size = test_size
                best_random_state = random_state

    return {'name': name, 'best_params': grid_search.best_params_, 'best_score': best_accuracy, 'test_size': best_test_size, 'random_state': best_random_state}

def get_model_instance(name, params=None):
    models = {
        "LR": LogisticRegression,
        "LDA": LinearDiscriminantAnalysis,
        "KNN": KNeighborsClassifier,
        "DT": DecisionTreeClassifier,
        "NB": GaussianNB,
        "SVM": SVC,
        "RF": RandomForestClassifier,
        "GB": GradientBoostingClassifier,
        "XGB": XGBClassifier,
        "LGBM": LGBMClassifier,
        "CatBoost": CatBoostClassifier,
        "MLP": MLPClassifier,
        "AdaBoost": AdaBoostClassifier,
        "Bagging": BaggingClassifier,
        "ExtraTrees": ExtraTreesClassifier,
        "GaussianProcess": GaussianProcessClassifier,
        "Ridge": Ridge,
        "Lasso": Lasso,
        "ElasticNet": ElasticNet
    }
    if params:
        return models[name](**params)
    else:
        return models[name]()

def get_parameter_grid(name):
    param_grids = {
        "LR": {'penalty': ['l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]},
        "LDA": {'solver': ['svd', 'lsqr', 'eigen']},
        "KNN": {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance', 'callable']},
        "DT": {'max_depth': [3, 5, 7, 9, 11], 'criterion': ['gini', 'entropy', 'mse', 'mae']},
        "NB": {},
        "SVM": {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid', 'laplacian', 'tanh']},
        "RF": {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7, 9, 11]},
        "GB": {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.5]},
        "XGB": {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.5]},
        "LGBM": {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.5]},
        "CatBoost": {'iterations': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.5]},
        "MLP": {'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)], 'activation': ['relu', 'tanh', 'sigmoid', 'softmax', 'leaky_relu', 'prelu', 'elu', 'swish', 'mish']},
        "AdaBoost": {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.5]},
        "Bagging": {'n_estimators': [50, 100, 200]},
        "ExtraTrees": {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7, 9, 11]},
        "GaussianProcess": {},
        "Ridge": {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]},
        "Lasso": {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]},
        "ElasticNet": {'alpha': [0.001, 0.01, 0.1, 1, 10, 100], 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}
    }
    return param_grids[name]


# Fonksiyonu çağır
#auto_ml(loaded_data)
def myself(df):
        if df is None:
            st.error('Please upload a dataset. If your dataset prepare for train and testing, go to next  step.', icon="🚨")

        else:
                df = pd.read_csv(uploaded_file)
                column_names = df.columns.tolist()
                hedef_degisken = st.selectbox("Hedef Değişkeni Seçin", column_names)

                # Define the list of models
                # Tüm modellerin listesi
                all_models = ["LR", "LDA", "KNN", "DT", "NB", "SVM", "RF", "GB", "XGB", "LGBM", "CatBoost", "MLP", "AdaBoost", "Bagging", "ExtraTrees", "GaussianProcess", "Ridge", "Lasso", "ElasticNet"]

                # Seçilen modellerin listesi
                selected_models = st.multiselect("Modelleri Seçin (Maksimum 5)", all_models, default=[])

                # Seçilen modellerin sayısı kontrol edilir
                if len(selected_models) > 5:
                        st.warning("Maksimum 5 model seçebilirsiniz!")
                        selected_models = selected_models[:5]  # Maksimum 5 modeli alır

                        # Seçilen modelleri göster
                        st.write("Seçilen Modeller:", selected_models)

                baslat_btn = st.button("Makine Öğrenmesini bu modeller üzerinden başlat.")
                if baslat_btn:
                       st.write("Makine öğrenmesi başlatıldı.")
                       auto_ml(df, hedef_degisken, selected_models)
              

#FRONTEND
if selected == "INPUT":
        # FRONTEND
        st.subheader("Machine Learning")
        st.write("Automates data cleaning Bir şirket, karar vermeBir şirket, kararBir şirket, karar vermeBir şirket, karar such as filling in missing data and deleting unnecessary columns. It also allows the user to perform basic pre-processing steps such as Label Encoding or One-Hot Encoding. Model Selection and Training: By presenting multiple machine learning models to the user, it evaluates the performance of each with different parameters and gives the user the chance to choose the best performing model.")
        with st.expander("🔗 Machine Learning"):
                st.write("Bir şirket, karar vermeBir şirket, karar verme sürecini şekillendirmek içinBir şirket, karar verme sürecini şekillendirmek içinBir şirket, karar verme sürecini şekillendirmek içinBir şirket, karar verme sürecini şekillendirmek için sürecini şekillendirmek için verileri kullanırken alakalı, eksiksiz ve doğru verileri kullanmaları çok önemlidir. Bununla birlikte, veri kümeleri genellikle analizden önce ortadan kaldırılması gereken hatalar içerir. Bu hatalar, tahminleri önemli ölçüde etkileyebilecek yanlış yazılmış tarihler, parasal değerler ve diğer ölçüm birimleri gibi biçimlendirme hatalarını içerebilir. Aykırı değerler, sonuçları her durumda çarpıttığından önemli bir endişe kaynağıdır. Yaygın olarak bulunan diğer veri hataları; bozuk veri noktalarını, eksik bilgileri ve yazım hatalarını içerir. Temiz veriler, ML modellerinin yüksek oranda doğru olmasına yardımcı olabilir. Düşük kaliteli eğitim veri kümelerini kullanmak, dağıtılan modellerde hatalı tahminlere neden olabileceğinden temiz ve doğru veriler, makine öğrenimi modellerini eğitmek için özellikle önemlidir. Veri bilimcilerinin, zamanlarının büyük bir kısmını makine öğrenimi için veri hazırlamaya ayırmalarının başlıca nedeni budur.")
                st.info('If you want to know more in detail, read our [**manual**](https://www.retmon.com/blog/veri-gorsellestirme-nedir#:~:text=Veri%20g%C3%B6rselle%C5%9Ftirme%3B%20verileri%20insan%20beyninin,i%C3%A7in%20kullan%C4%B1lan%20teknikleri%20ifade%20eder.).', icon="ℹ️")
        
        # Dosya yükleme işlemi için Streamlit'in file_uploader fonksiyonunu kullanma
        uploaded_file = st.file_uploader("Upload a file", type=["csv", "yaml", "json", "jpg", "png"])

        tab1, tab2= st.tabs(["INPUTS", "EXAMPLE"])
        with tab1:
                        with st.container(border=True):
                                bitirme = st.radio(
                                        "Set label visibility 👇",
                                        ["I'll do it myself.", "Do it."],
                                        key="visibility",
                                        )
                                
                                if bitirme == "I'll do it myself.":
                                      #myself(uploaded_file)
                                      st.write("Fonksiyon güncellenecek.")
                                      pass
                                else:
                                      myself(uploaded_file)
                                
                                
        #Örnek Kod Uygulaması                  
        with tab2:
                        with open("media/texts/randomForest.txt", "r") as file:
                                code = file.read()
                        st.code(code, language='python')
                        

                        
else: 
        st.write("dsdsd")
