import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_card import card
from streamlit_lottie import st_lottie  # pip install streamlit-lottie
import pandas as pd
import yaml, json, requests, base64
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import os, zipfile
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

def scaling(df, choose):
        if choose == "Min-Max Scaling":
                # Min-Max Scaling
                min_max_scaler = MinMaxScaler()
                df_min_max_scaled = pd.DataFrame(min_max_scaler.fit_transform(df), columns=df.columns)
                return df_min_max_scaled
        elif choose == "Z-Score Scaling":
                # Z-Score Scaling
                standard_scaler = StandardScaler()
                df_standard_scaled = pd.DataFrame(standard_scaler.fit_transform(df), columns=df.columns)
                return df_standard_scaled
        elif choose == "Robust Scaling":
                # Robust Scaling
                robust_scaler = RobustScaler()
                df_robust_scaled = pd.DataFrame(robust_scaler.fit_transform(df), columns=df.columns)
                return df_robust_scaled
        elif choose == "MaxAbs Scaling":
                # MaxAbs Scaling
                maxabs_scaler = MaxAbsScaler()
                df_maxabs_scaled = pd.DataFrame(maxabs_scaler.fit_transform(df), columns=df.columns)
                return df_maxabs_scaled
        else:
               st.write("Geçersiz seçim")
               return df



#Animasyon ekleme
def load_lottiefile(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

#Loading animasyonu
lottie_coding = load_lottiefile("media/lottieFiles/loading.json")  # replace link to local lottie file

#Temizlenen dosyası indirme
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")

def save_uploaded_file(uploadedfile):
    with open(os.path.join("tempDir", uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved file :{} in tempDir".format(uploadedfile.name))

#Sekme 1 fonksiyonu: İlk ve son 10 satırı gösterir.
def row_of_datasets(df):
        st.write("**First 10 Rows**")
        st.write(df.head(10))  # İlk 10 satırı gösterme
        st.write("**Last 10 Rows**")
        st.write(df.tail(10))  # Son 10 satırı gösterme

#Sekme 2: Sayısal ve genel bilgileri tablolar aracılığıyla gösterir.
def information(df):
         # Bilgi açıklaması
        st.caption("The table below shows the number of rows, columns, empty columns, and column names of the dataset.")

        # Satır ve sütun sayısı, sütun isimleri ve boş hücre sayısını gösterir
        data = pd.DataFrame({
                "Information": ["Row", "Columns", "Column Names", "Empty Values"],
                "Value": [df.shape[0], df.shape[1], df.columns.tolist(), df.isnull().sum().sum()]
        })
        st.dataframe(data)

        # Sütunların türlerini görme ve beş satır örnek gösterme
        col1, col2 = st.columns([0.4, 0.6])
        with col1:
                st.write("**Data Types of Columns**")
                st.write(df.dtypes)
        with col2:
                st.write("**From which column would you like to see sample values?**")
                selected_column = st.selectbox("Column Names:", df.columns.tolist())
                st.write(df[selected_column].head())
        
        # Kategorik sütunların istatistikleri
        if any(df.dtypes == 'object'):
                st.write("**Statistics of Categorical Columns**")
                st.write(df.describe(include=['O']))
        else:
                st.info('No categorical data available.', icon="ℹ️")

        # Sayısal sütunların istatistikleri
        if any(df.dtypes != 'object'):
                st.write("**Statistics of Numerical Columns**")
                st.write(df.describe())
        else:
                st.info('No numerical data available.', icon="ℹ️")

def save_uploaded_file(uploadedfile):
    with open(os.path.join("tempDir", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success(f"Saved file: {uploadedfile.name} in tempDir")

def analyze_image_folders(base_dir):
    folder_data = []
    
    for root, dirs, files in os.walk(base_dir):
        folder_info = {
            "folder_name": os.path.basename(root),
            "file_count": len(files),
            "files": files,
            "file_types": {}
        }
        
        for file in files:
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension in folder_info["file_types"]:
                folder_info["file_types"][file_extension] += 1
            else:
                folder_info["file_types"][file_extension] = 1
        
        folder_data.append(folder_info)
    
    return folder_data

def display_analysis(base_dir, folder_data):
    folder_info = []
    for folder in folder_data:
        folder_name = folder['folder_name']
        file_count = folder['file_count']
        files = ', '.join(folder['files'][:20]) if file_count > 20 else ', '.join(folder['files'])
        file_types = folder['file_types']

        folder_info.append([folder_name, file_count, files, file_types])

    df = pd.DataFrame(folder_info, columns=['Klasör Adı', 'Dosya Sayısı', 'İlk 20 Dosya Adı', 'Dosya Türleri ve Sayıları'])

    # Ek analizler
    for folder in folder_data:
        folder_path = os.path.join(base_dir, folder["folder_name"])
        file_count = folder['file_count']
        if file_count > 20:
            st.write(f"--- {folder['folder_name']} ---")
            st.write(f"İlk 20 dosya adı: {', '.join(folder['files'][:20])}")
            st.write("Dosya türleri ve sayıları:")
            for file_type, count in folder["file_types"].items():
                st.write(f"  - {file_type}: {count}")
            st.write(f"Toplam dosya sayısı: {file_count}")
            st.write("Analizler:")

    st.table(df)

def display_sample_images(base_dir, folder_data):
        try:
                st.write("Örnek Görseller:")
                for folder in folder_data:
                        folder_path = os.path.join(base_dir, folder["folder_name"])
                        sample_files = folder["files"][:3]  # İlk 3 dosyayı seç
                        for file in sample_files:
                                file_path = os.path.join(folder_path, file)
                                if is_valid_image_file(file_path):
                                        image = Image.open(file_path)
                                        st.image(image, caption=file)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
                st.error('Yüklenen veri seti boş veya geçersiz.', icon="🚨")
        except Exception as e:
                st.error(f'Bir hata oluştu: {e}', icon="🚨")

def is_valid_image_file(file_path):
    valid_extensions = ['.jpg', '.jpeg', '.png']
    return any(file_path.lower().endswith(ext) for ext in valid_extensions)

def plot_image_statistics(folder_data):
    folder_names = [folder["folder_name"] for folder in folder_data]
    file_counts = [folder["file_count"] for folder in folder_data]
    
    # Bar grafiği
    plt.figure(figsize=(10, 6))
    plt.bar(folder_names, file_counts, color='skyblue')
    plt.xlabel('Klasör Adı')
    plt.ylabel('Dosya Sayısı')
    plt.title('Klasörlerdeki Dosya Sayısı')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

    # Pasta grafiği
    plt.figure(figsize=(8, 8))
    plt.pie(file_counts, labels=folder_names, autopct='%1.1f%%', startangle=140)
    plt.title('Klasörlerdeki Dosya Dağılımı')
    plt.axis('equal')
    plt.tight_layout()
    st.pyplot(plt)

def prepare_data(df, operations, deleted, missing, normalize_columns, scaling):
    islemler = []
    st.write("Yapılacak işlemler:", operations)
    newdata = df.copy()

    if "Remove Duplicate Rows" in operations: 
        newdata.drop_duplicates(inplace=True)
        islemler.append("Yinelenen satırlar silindi.")
    else:
        islemler.append("Yinelenen satır bulunmuyor. Değişiklik yapılmadı.")

    if "Lowercase Column Names" in operations: 
        newdata.columns = [kolon.lower() for kolon in newdata.columns]
        islemler.append("Sütun isimleri küçük harfe çevrildi.")
    else:
        islemler.append("Tüm sutün isimleri küçük harfle başlıyor. Değişiklik yapılmadı.")

    if "Lowercase Values in Object Data" in operations: 
        object_columns = newdata.select_dtypes(include=['object']).columns
        newdata[object_columns] = newdata[object_columns].apply(lambda x: x.astype(str).str.lower())
        islemler.append("Object veri tipinde olan sütunlardaki değerler küçük harfe dönüştürüldü.")
    else:
        islemler.append("Object veri tipinde olan sütunlardaki değerler küçük harf. Değişiklik yapılmadı.")

    if "Delete Unnecessary Columns" in operations and deleted: 
        newdata.drop(columns=deleted, inplace=True)
        islemler.append("Gereksiz kolonlar silindi.")
        
    if "Fill Missing Values with -Unknown- in String Values" in operations: 
        string_columns = newdata.select_dtypes(include=['object']).columns
        newdata[string_columns] = newdata[string_columns].fillna("Unknown")
        islemler.append("String değerlere 'Bilinmiyor' ile eksik değerler dolduruldu.")
    else:
        islemler.append("Boş bir String değer bulunmuyor. Değişiklik yapılmadı.")


    if "Label Encoding" in operations: 
        label_encoder = LabelEncoder()
        string_columns = newdata.select_dtypes(include=['object']).columns
        newdata[string_columns] = newdata[string_columns].apply(label_encoder.fit_transform)
        islemler.append("Label Encoding işlemi tamamlandı.")
           
    if "One-Hot Encoding" in operations:
        newdata = pd.get_dummies(newdata)
        islemler.append("One-Hot Encoding işlemi tamamlandı.")
 
    if "Fill Missing Values in Selected Columns" in operations:
        missing_columns = newdata.columns[newdata.isnull().any()].tolist()
        if not missing_columns:
            st.write("Veri setinde eksik değer içeren sütun bulunmuyor.")
        else:
            for column in missing:
                if newdata[column].isnull().any():
                    if newdata[column].dtype in ['int64', 'float64']:
                        imputer = SimpleImputer(strategy='mean')
                        newdata[[column]] = imputer.fit_transform(newdata[[column]])
                        st.write(f"{column} sütunundaki eksik değerler ortalama ile dolduruldu.")

    if "Normalize" in operations:
                scaler = MinMaxScaler()
                df[normalize_columns] = scaler.fit_transform(df[normalize_columns])
                return df
                islemler.append("Seçilen sütunlar normalize edildi.")
    else:
               pass

    if "Scaling" in operations:
                if scaling == "Min-Max Scaling":
                        # Min-Max Scaling
                        min_max_scaler = MinMaxScaler()
                        newdata = pd.DataFrame(min_max_scaler.fit_transform(df), columns=df.columns)
                elif scaling == "Z-Score Scaling":
                        # Z-Score Scaling
                        standard_scaler = StandardScaler()
                        newdata = pd.DataFrame(standard_scaler.fit_transform(df), columns=df.columns)
                elif scaling == "Robust Scaling":
                        # Robust Scaling
                        robust_scaler = RobustScaler()
                        newdata = pd.DataFrame(robust_scaler.fit_transform(df), columns=df.columns)
                elif scaling == "MaxAbs Scaling":
                        # MaxAbs Scaling
                        maxabs_scaler = MaxAbsScaler()
                        newdata = pd.DataFrame(maxabs_scaler.fit_transform(df), columns=df.columns)
                else:
                        st.write("Geçersiz seçim")
                islemler.append("Scaling işlemi yapıldı.")
    else:
               pass

    # Veri setini CSV formatında indirme işlemi
    csv = convert_df(newdata)
    st.download_button(
        label="Download Data",
        data=csv,
        file_name="updatedDataset.csv",
        mime="text/csv",)
    
    st.write(islemler)

def auto_cleaning(df):
        op = ["Remove Duplicate Rows", "Lowercase Column Names", "Lowercase Values in Object Data" "Fill Missing Values with -Unknown- in String Values", "One-Hot Encoding"]
        prepare_data(df, op, [], [], [], [])

def myself_cleaning(df):
        with st.container(border = True):
                delete_columns = []
                missing_columns = []
                normalize_columns = []
                scaling_type = []
                op = ["Remove Duplicate Rows", #Yinelenen satırları kaldırma 
                        "Lowercase Column Names",  #Sütun isimlerini küçük harfe çevirme
                        "Lowercase Values in Object Data," #Kategorik verileri küçük harfe çevirme
                        "Delete Unnecessary Columns", #Gereksiz kolonları silme
                        "Fill Missing Values with -Unknown- in String Values", 
                        "Label Encoding", 
                        "One-Hot Encoding", 
                        "Fill Missing Values in Selected Columns", 
                        "Normalize Selected Columns", "Scaling Data"]
                                        
                operations = st.multiselect("Choose cleaning operations:", op)
                if "Delete Unnecessary Columns" in operations:
                        delete_columns = st.multiselect("Silinmesini istediğiniz sütunları seçin: ", df.columns.tolist())
                if "Find Columns with Missing Values" in operations:
                        missing_columns = st.multiselect("Eksik değerleri doldurmak istediğiniz kolon isimleri", df.columns.tolist())
                if "Normalize Selected Columns" in operations:
                        normalize_columns = st.multiselect("Normalize etmek istediğiniz kolon isimleri", df.columns.tolist())
                if "Scaling Data" in operations:
                        scaling_type = st.selectbox("Hangi scaling'i uygulamak istersiniz?", ["Min-Max Scaling", "Z-Score Scaling", "Robust Scaling", "MaxAbs Scaling"])
                
                
                if st.button("Clean"):
                        prepare_data(df, operations, delete_columns, missing_columns, normalize_columns, scaling_type)
                

#### FRONTEND
def frontend():
        st.header("Data Set Cleaning and Editing")
        st.write("Automates data cleaning steps such as filling in missing data and deleting unnecessary columns. It also allows the user to perform basic pre-processing steps such as Label Encoding or One-Hot Encoding. Model Selection and Training: By presenting multiple machine learning models to the user, it evaluates the performance of each with different parameters and gives the user the chance to choose the best performing model.")
        with st.expander("👀 Data Analysis"):
                st.write("Data analysis is the process of transforming raw data into meaningful information. It starts with collecting, cleaning, exploring and visualizing data. This is followed by data transformation and modeling. Analysis results are interpreted and reported and contribute to decision-making processes. Programming languages such as Python and R offer powerful tools for data analysis. Accurate data analysis helps businesses make strategic decisions and improve performance.")
        with st.expander("🔗 Data Cleaning"):
                st.write("Data cleaning is the process of making raw data analyzable. It involves identifying and appropriately filling in missing data, correcting erroneous or inconsistent data, and identifying and addressing outliers. Data cleaning improves the accuracy and reliability of analysis results. Programming languages such as Python and R offer powerful tools and libraries for data cleaning. Proper data cleansing ensures the quality and reliability of the information businesses get from data analysis.")
        with st.expander("🔍 Dataset Search"):
                st.subheader("Streamlit Search")
                st.write("If you don't have a dataset or want a new one, you can use the dataset search.")
                # Excel dosyasını yükleyin
                excel_file = 'media/datasets/advice_datasets.xlsx'
                df = pd.read_excel(excel_file)

                # Arama kutusu
                search_query = st.text_input("🔎 Search for a Dashboard", placeholder="Search...")

                if search_query:
                        # Arama terimini küçük harflere çevir
                        search_query = search_query.lower()
                        
                        # DataFrame içinde arama yap
                        results = df[df['DATASET NAME'].str.contains(search_query, case=False)]
                        
                        if not results.empty:
                                st.write(f"{len(results)} sonuç bulundu:")
                                for index, row in results.iterrows():
                                        st.write(f"**{row['DATASET NAME']}**: {row['LINK']}")
                        else:
                                st.write("Aradığınız anahtar kelimeye uygun bir veri seti bulunamadı. ")
                        
        st.info('If you want to know more in detail, read our [**manual**](https://www.retmon.com/blog/veri-gorsellestirme-nedir#:~:text=Veri%20g%C3%B6rselle%C5%9Ftirme%3B%20verileri%20insan%20beyninin,i%C3%A7in%20kullan%C4%B1lan%20teknikleri%20ifade%20eder.).', icon="ℹ️")

        # Dosya yükleme işlemi için Streamlit'in file_uploader fonksiyonunu kullanma
        uploaded_file = st.file_uploader("Upload a file", type=["csv", "yaml", "json", "zip"])
        st.caption("You can upload your dataset in CSV, YAML, JSON and ZIP formats. The process is carried out by converting csv, yaml and json files to cv format. ZIP files are image files. Make sure that your ZIP file does not contain text or other contradictory files.")

        #Dosya türünü analiz eder ve dosya türü json veya yaml ise csv dosyasına çevirir.
        #Analiz ve temizleme işlemleri sekmeli sayfada yapılır.
        tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "Information", "Graphics", "Cleaning"])                    
        with tab1:
                try:
                        if uploaded_file is None:
                                st.error('Lütfen bir veri seti yükleyin. Veri setinizi eğitim ve test için hazırladıysanız, bir sonraki adıma geçin.', icon="🚨")
                        else:
                                if uploaded_file.type == 'text/csv':  # CSV dosyası yüklenirse
                                        df = pd.read_csv(uploaded_file)
                                        if df.empty:
                                                st.error('Yüklenen veri seti boş.', icon="🚨")
                                        else:
                                                row_of_datasets(df)

                                elif uploaded_file.type == 'text/yaml':  # YAML dosyası yüklenirse
                                        yaml_verisi = yaml.safe_load(uploaded_file)
                                        # YAML verisini DataFrame'e dönüştür
                                        df = pd.DataFrame(yaml_verisi)
                                        # CSV dosyasına kaydet
                                        df.to_csv("veri.csv", index=False)
                                        # CSV dosyasını tekrar yükle
                                        df = pd.read_csv("veri.csv")
                                        row_of_datasets(df)

                                elif uploaded_file.type == 'application/json':  # JSON dosyası yüklenirse
                                        json_verisi = json.load(uploaded_file)
                                        # JSON verisini DataFrame'e dönüştür
                                        df = pd.DataFrame(json_verisi)
                                        # CSV dosyasına kaydet
                                        df.to_csv("veri.csv", index=False)
                                        # CSV dosyasını tekrar yükle
                                        df = pd.read_csv("veri.csv")
                                        row_of_datasets(df)

                                elif uploaded_file.type in ['application/zip', 'application/x-zip-compressed']:  # Zip dosyası yüklenirse (resim klasörü)
                                        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                                                zip_ref.extractall("extracted_images")
                                        st.success('Resim klasörü başarıyla yüklendi ve açıldı.')

                                        folder_data = analyze_image_folders("extracted_images")
                                        #display_analysis(folder_data)
                                        display_sample_images("extracted_images", folder_data)
                                else:
                                        st.write("Dosya formatı desteklenmiyor.")


                except (pd.errors.EmptyDataError, pd.errors.ParserError):

                        st.error('Yüklenen veri seti boş veya geçersiz.', icon="🚨")
                except Exception as e:
                        st.error(f'Bir hata oluştu: {e}', icon="🚨")
        with tab2:
                try:
                        if uploaded_file is None:
                                st.error('Please upload a dataset. If your dataset prepare for train and testing, go to next  step.', icon="🚨")
                        
                        else:
                                if uploaded_file.type in ['application/zip', 'application/x-zip-compressed']:  # Zip dosyası ise
                                        folder_data = analyze_image_folders("extracted_images")
                                        display_analysis(folder_data)
                                elif df.empty:
                                        st.error('The uploaded dataset is empty.', icon="🚨")
                                else:
                                        information(df)
                except (pd.errors.EmptyDataError, pd.errors.ParserError):
                        st.error('Yüklenen veri seti boş veya geçersiz.', icon="🚨")
                except Exception as e:
                        st.error(f'Bir hata oluştu: {e}', icon="🚨")
        with tab3:
                try:
                        if uploaded_file is None:
                                st.error('Please upload a dataset. If your dataset prepare for train and testing, go to next  step.', icon="🚨")
                        else:
                                if uploaded_file.type in ['application/zip', 'application/x-zip-compressed']:  # Zip dosyası ise
                                        folder_data = analyze_image_folders("extracted_images")
                                        plot_image_statistics(folder_data)
                                else:
                                        gr = st.selectbox("Hangi grafiği görüntülemek istiyorsun?", ["Line Chart", "Bar Chart", "Area Chart", "Scatter Chart"])
                                        column_c = st.multiselect("Grafiğe dönüştürmek istediğiniz sütunları seçin.", df.columns.tolist() )
                                        if gr == "Line Chart":
                                                st.caption("Line chart, verilerin bir çizgi grafiği üzerinde gösterildiği bir grafik türüdür. X ekseni genellikle zaman veya kategorik değerlerle ilişkilendirilirken, Y ekseni ise bu değerlerin karşılık geldiği ölçülebilir verileri temsil eder. Her veri noktası, X ekseni boyunca konumlandırılır ve belirli bir Y ekseni değerine sahiptir. Bu noktalar, birbirlerine doğrusal olarak bağlanarak bir çizgi oluşturulur. Line chart, veri setindeki değişiklikleri izlemek, trendleri analiz etmek ve ilişkileri anlamak için kullanılır. Özellikle zaman serileri verileriyle çalışırken, belirli bir zaman aralığındaki değişimleri anlamak için line chart çok kullanışlıdır.")
                                                chart_data = pd.DataFrame(np.random.randn(20, len(column_c)), columns=column_c)
                                                st.line_chart(chart_data)
                                        elif gr == "Bar Chart":
                                                st.caption("Line chart, verilerin bir çizgi grafiği üzerinde gösterildiği bir grafik türüdür. X ekseni genellikle zaman veya kategorik değerlerle ilişkilendirilirken, Y ekseni ise bu değerlerin karşılık geldiği ölçülebilir verileri temsil eder. Her veri noktası, X ekseni boyunca konumlandırılır ve belirli bir Y ekseni değerine sahiptir. Bu noktalar, birbirlerine doğrusal olarak bağlanarak bir çizgi oluşturulur. Line chart, veri setindeki değişiklikleri izlemek, trendleri analiz etmek ve ilişkileri anlamak için kullanılır. Özellikle zaman serileri verileriyle çalışırken, belirli bir zaman aralığındaki değişimleri anlamak için line chart çok kullanışlıdır.")
                                                chart_data = pd.DataFrame(np.random.randn(30, len(column_c)), columns=column_c)
                                                st.bar_chart(chart_data)
                                        elif gr == "Area Chart":
                                                st.caption("Line chart, verilerin bir çizgi grafiği üzerinde gösterildiği bir grafik türüdür. X ekseni genellikle zaman veya kategorik değerlerle ilişkilendirilirken, Y ekseni ise bu değerlerin karşılık geldiği ölçülebilir verileri temsil eder. Her veri noktası, X ekseni boyunca konumlandırılır ve belirli bir Y ekseni değerine sahiptir. Bu noktalar, birbirlerine doğrusal olarak bağlanarak bir çizgi oluşturulur. Line chart, veri setindeki değişiklikleri izlemek, trendleri analiz etmek ve ilişkileri anlamak için kullanılır. Özellikle zaman serileri verileriyle çalışırken, belirli bir zaman aralığındaki değişimleri anlamak için line chart çok kullanışlıdır.")
                                                chart_data = pd.DataFrame(np.random.randn(20, len(column_c)), columns=column_c)
                                                st.area_chart(chart_data)
                                        elif gr == "Scatter Chart":
                                                st.caption("Line chart, verilerin bir çizgi grafiği üzerinde gösterildiği bir grafik türüdür. X ekseni genellikle zaman veya kategorik değerlerle ilişkilendirilirken, Y ekseni ise bu değerlerin karşılık geldiği ölçülebilir verileri temsil eder. Her veri noktası, X ekseni boyunca konumlandırılır ve belirli bir Y ekseni değerine sahiptir. Bu noktalar, birbirlerine doğrusal olarak bağlanarak bir çizgi oluşturulur. Line chart, veri setindeki değişiklikleri izlemek, trendleri analiz etmek ve ilişkileri anlamak için kullanılır. Özellikle zaman serileri verileriyle çalışırken, belirli bir zaman aralığındaki değişimleri anlamak için line chart çok kullanışlıdır.")
                                                chart_data = pd.DataFrame(np.random.randn(30, len(column_c)), columns=column_c)
                                                st.scatter_chart(chart_data)
                                        else:
                                                pass
                except (pd.errors.EmptyDataError, pd.errors.ParserError):
                        st.error('Yüklenen veri seti boş veya geçersiz.', icon="🚨")
                except Exception as e:
                        st.error(f'Bir hata oluştu: {e}', icon="🚨")              
        with tab4:
                try:
                        if uploaded_file is None:
                                st.error('Please upload a dataset. If your dataset prepare for train and testing, go to next  step.', icon="🚨")
                        else:
                                cleaning = st.selectbox("Do you want to clean the data yourself or have it cleaned automatically?", ["Choose", "Auto Dataset Cleaning", "Cleaning the Dataset Yourself"])

                
                                if cleaning == "Auto Dataset Cleaning":
                                        auto_cleaning(df)
                                elif cleaning == "Cleaning the Dataset Yourself":
                                        myself_cleaning(df)
                                else:
                                        st.write("Please select a valid data clearing option.")

                                
                except (pd.errors.EmptyDataError, pd.errors.ParserError):
                        st.error('Yüklenen veri seti boş veya geçersiz.', icon="🚨")
                except Exception as e:
                        st.error(f'Bir hata oluştu: {e}', icon="🚨")
                                
if __name__ == "__main__":
       # Sayfa adını ve sayfanın geniş olmasını sağlama
        st.set_page_config(page_title="ATOM AI", layout="wide", initial_sidebar_state="expanded")
        
        frontend()

        