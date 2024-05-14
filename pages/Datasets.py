import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_card import card
from streamlit_lottie import st_lottie  # pip install streamlit-lottie
import pandas as pd
import yaml, json, requests
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import base64

# Sayfa adını ve sayfanın geniş olmasını sağlama
st.set_page_config(
    page_title="ATOM AI",
    layout="wide",
    initial_sidebar_state="expanded"

)

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

#Sekme 1 fonksiyonu: İlk ve son 10 satırı gösterir.
def row_of_datasets(df):
        st.write("**First 10 Rows**")
        st.write(df.head(10))  # İlk 10 satırı gösterme
        st.write("**Last 10 Rows**")
        st.write(df.tail(10))  # Son 10 satırı gösterme

#Sekme 2: Sayısal ve genel bilgileri tablolar aracılığıyla gösterir.
def information(df):
        st.caption("The table below shows the number of rows, columns, empty columns and column names of the dataset.The table below shows the number of rows, columns, empty columns and column names of the dataset.The table below shows the number of rows, columns, empty columns and column names of the dataset.The table below shows the number of rows, columns, empty columns and column names of the dataset.")
        #Satır ve sütun sayısı, sütun isimleri ve boş hücre sayısını gösterir.
        data = pd.DataFrame({
                        "Information": ["Row", "Columns", "Column Names", "Empty Values"],
                        "Value": [df.shape[0], df.shape[1], df.columns.tolist(), df.isnull().sum().sum() ]
                })
        st.dataframe(data)

        #Sütunların türlerini görme ve beş satır örnek gösterme
        col1, col2 = st.columns([0.4, 0.6])
        with col1:
                st.write("**Data Types of Columns**")
                st.write(df.dtypes)
        with col2: 
                st.write("**From which column would you like to see sample values?**")
                hel = st.selectbox("Column Names:", df.columns.tolist())
                st.write(df[hel].iloc[:5])
        try:
                st.write("**Statistics of Categorical Columns**")
                st.write(df.describe(include=['O']))
        except: 
                st.info('No categorical data available.', icon="ℹ️")

        
        try:
                st.write("**Statistics of Numerical Columns**")
                st.write(df.describe())
        except:
                st.info('No numerical data available.', icon="ℹ️")

def prepare_data(df, operation, deleted, missing):
    op = ["Remove Duplicate Rows", #Yinelenen satırları kaldırma 
                                      "Lowercase Column Names",  #Sütun isimlerini küçük harfe çevirme
                                      "Lowercase Values in Object Data," #Kategorik verileri küçük harfe çevirme
                                      "Delete Unnecessary Columns", #Gereksiz kolonları silme
                                      "Fill Missing Values with -Unknown- in String Values", 
                                      "Label Encoding", 
                                      "One-Hot Encoding", 
                                      "Fill Missing Values in Selected Columns"]
    islemler = []
    st.write("Yapılacak işlemler", operation)
    newdata = df
    
    if op[0] in operations: 
        # Yinelenen satırları kaldırma
        newdata.drop_duplicates(inplace=True)
        islemler.append("Yinelenen satırlar silindi.")

    if op[1] in operations: 
        # Sütun adlarını küçük harfe dönüştürme
        newdata.columns = [kolon.lower() for kolon in newdata.columns]
        islemler.append("Sütun isimleri küçük harfe çevirdi.")

    if op[2] in operations: 
        # Sadece object veri tipinde olan sütunlardaki değerleri küçük harfe dönüştürme
        object_columns = newdata.select_dtypes(include=['object']).columns
        newdata[object_columns] = newdata[object_columns].apply(lambda x: x.astype(str).str.lower())
        islemler.append("Object veri tipinde olan sütunlardaki değerleri küçük harfe dönüştürüldü.")

    if op[3] in operations: 
        # Gereksiz kolonları silme
        newdata = newdata.drop(columns=deleted)
        islemler.append("Kolonlar silindi.")
        
    if op[4] in operations: 
        # String değerlere "Bilinmiyor" ile eksik değerleri doldurma
        string_columns = newdata.select_dtypes(include=['object']).columns
        newdata[string_columns] = newdata[string_columns].fillna("Unknown")
        islemler.append("String değerlere 'Bilinmiyor' ile eksik değerleri doldurma")

    if op[5] in operations: 
            # Label Encoding işlemi burada gerçekleşecek
            label_encoder = LabelEncoder()
            string_columns = newdata.select_dtypes(include=['object']).columns
            newdata[string_columns] = newdata[string_columns].apply(label_encoder.fit_transform)
            islemler.append("Label Encoding işlemi tamamlandı.")
           
    if op[6] in operations:
            # One-Hot Encoding işlemi burada yapılacak
            newdata = pd.get_dummies(df)
            islemler.append("One-Hot Encoding işlemi tamamlandı.")
 
    if op[7] in operations:
        # Eksik değerlere sahip sütunları bulma
        sutunlar_eksik_deger = newdata.columns[df.isnull().any()].tolist()

        # Eğer eksik değer içeren bir sütun yoksa
        if not sutunlar_eksik_deger:
                st.write("Veri setinde eksik değer içeren sütun bulunmuyor.")
        else:
                while True:
                        secim = st.multiselect("Eksik değerleri doldurmak istediğiniz kolon isimleri", df.columns.tolist())
                        
                        if secim is not None:
                                secilen_sutunlar = secim
                                # Kullanıcının girdiği sütunların doğruluğunu kontrol etme
                                hatali_sutunlar = [s for s in secilen_sutunlar if s not in newdata.columns]
                                if hatali_sutunlar:
                                        print(f"Geçersiz sütunlar: {', '.join(hatali_sutunlar)}")
                                else:
                                        break
    
                # Eksik değerlere sahip sütunları bulma
                sutunlar_eksik_deger = df.columns[df.isnull().any()].tolist()

                # Eğer eksik değer içeren bir sütun yoksa
                if not sutunlar_eksik_deger:
                        print("Veri setinde eksik değer içeren sütun bulunmuyor.")
                else:
                        while True:
                                secim = missing
                                # Seçilen sütunlardaki eksik değerleri doldurma işlemi
                                for kolon in secim:
                                        if newdata[kolon].isnull().any():
                                                if newdata[kolon].dtype in ['int64', 'float64']:
                                                        imputer = SimpleImputer(strategy='mean')
                                                        newdata[[kolon]] = imputer.fit_transform(df[[kolon]])
                                                        st.write(f"{kolon} sütunundaki eksik değerler ortalama ile dolduruldu.")
                                                else:
                                                        st.write(f"{kolon} sütunu sayısal bir veri tipine sahip değil, eksik değerler doldurulmadı.")

    # Veri setini CSV formatında indirme işlemi
    csv = convert_df(newdata)
    st.download_button(
        label="Download Data",
        data=csv,
        file_name="updatedDataset.csv",
        mime="text/csv",
     )
    
    st.write(islemler)

    

        
# FRONTEND
st.header("Data Set Cleaning and Editing")
st.write("Automates data cleaning steps such as filling in missing data and deleting unnecessary columns. It also allows the user to perform basic pre-processing steps such as Label Encoding or One-Hot Encoding. Model Selection and Training: By presenting multiple machine learning models to the user, it evaluates the performance of each with different parameters and gives the user the chance to choose the best performing model.")
with st.expander("🔗 Data Analysis"):
        st.write("Bir şirket, karar vermeBir şirket, karar verme sürecini şekillendirmek içinBir şirket, karar verme sürecini şekillendirmek içinBir şirket, karar verme sürecini şekillendirmek içinBir şirket, karar verme sürecini şekillendirmek için sürecini şekillendirmek için verileri kullanırken alakalı, eksiksiz ve doğru verileri kullanmaları çok önemlidir. Bununla birlikte, veri kümeleri genellikle analizden önce ortadan kaldırılması gereken hatalar içerir. Bu hatalar, tahminleri önemli ölçüde etkileyebilecek yanlış yazılmış tarihler, parasal değerler ve diğer ölçüm birimleri gibi biçimlendirme hatalarını içerebilir. Aykırı değerler, sonuçları her durumda çarpıttığından önemli bir endişe kaynağıdır. Yaygın olarak bulunan diğer veri hataları; bozuk veri noktalarını, eksik bilgileri ve yazım hatalarını içerir. Temiz veriler, ML modellerinin yüksek oranda doğru olmasına yardımcı olabilir. Düşük kaliteli eğitim veri kümelerini kullanmak, dağıtılan modellerde hatalı tahminlere neden olabileceğinden temiz ve doğru veriler, makine öğrenimi modellerini eğitmek için özellikle önemlidir. Veri bilimcilerinin, zamanlarının büyük bir kısmını makine öğrenimi için veri hazırlamaya ayırmalarının başlıca nedeni budur.")

with st.expander("🔗 Data Cleaning"):
        st.write("Bir şirket, karar vermeBir şirket, karar verme sürecini şekillendirmek içinBir şirket, karar verme sürecini şekillendirmek içinBir şirket, karar verme sürecini şekillendirmek içinBir şirket, karar verme sürecini şekillendirmek için sürecini şekillendirmek için verileri kullanırken alakalı, eksiksiz ve doğru verileri kullanmaları çok önemlidir. Bununla birlikte, veri kümeleri genellikle analizden önce ortadan kaldırılması gereken hatalar içerir. Bu hatalar, tahminleri önemli ölçüde etkileyebilecek yanlış yazılmış tarihler, parasal değerler ve diğer ölçüm birimleri gibi biçimlendirme hatalarını içerebilir. Aykırı değerler, sonuçları her durumda çarpıttığından önemli bir endişe kaynağıdır. Yaygın olarak bulunan diğer veri hataları; bozuk veri noktalarını, eksik bilgileri ve yazım hatalarını içerir. Temiz veriler, ML modellerinin yüksek oranda doğru olmasına yardımcı olabilir. Düşük kaliteli eğitim veri kümelerini kullanmak, dağıtılan modellerde hatalı tahminlere neden olabileceğinden temiz ve doğru veriler, makine öğrenimi modellerini eğitmek için özellikle önemlidir. Veri bilimcilerinin, zamanlarının büyük bir kısmını makine öğrenimi için veri hazırlamaya ayırmalarının başlıca nedeni budur.")

st.info('If you want to know more in detail, read our [**manual**](https://www.retmon.com/blog/veri-gorsellestirme-nedir#:~:text=Veri%20g%C3%B6rselle%C5%9Ftirme%3B%20verileri%20insan%20beyninin,i%C3%A7in%20kullan%C4%B1lan%20teknikleri%20ifade%20eder.).', icon="ℹ️")

# Dosya yükleme işlemi için Streamlit'in file_uploader fonksiyonunu kullanma
uploaded_file = st.file_uploader("Upload a file", type=["csv", "yaml", "json", "jpg", "png"])

st.caption("Automates data cleaning steps such as filling in missing data and deleting unnecessary columns. It also allows the user to perform basic pre-processing steps such as Label Encoding or One-Hot Encoding. Model Selection and Training: By presenting multiple machine learning models to the user, it evaluates the performance of each with different parameters and gives the user the chance to choose the best performing model.")

#Dosya türünü analiz eder ve dosya türü json veya yaml ise csv dosyasına çevirir.
#Analiz ve temizleme işlemleri sekmeli sayfada yapılır.
tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "Information", "Graphics", "Cleaning"])
                      
with tab1:
        try:
                if uploaded_file is None:
                        st.error('Lütfen bir veri seti yükleyin. Veri setinizi eğitim ve test için hazırladıysanız, bir sonraki adıma geçin.', icon="🚨")
                elif uploaded_file.type == 'text/csv':  # CSV dosyası yüklenirse
                        df = pd.read_csv(uploaded_file)
                        if df.empty:
                                st.error('Yüklenen veri seti boş.', icon="🚨")
                                df = pd.read_csv(uploaded_file)
                        else:
                                row_of_datasets(df)

                elif uploaded_file.type == 'text/yaml': # YAML dosyası yüklenirse
                        yaml_verisi = yaml.safe_load(uploaded_file)
                        # YAML verisini DataFrame'e dönüştür
                        df = pd.DataFrame(yaml_verisi)
                        # CSV dosyasına kaydet
                        df.to_csv("veri.csv", index=False)
                        df = "veri.csv"
                elif uploaded_file.type == 'text/json':  # Json dosyası yüklenirse
                        json_verisi = json.load(uploaded_file)
                        # JSON verisini DataFrame'e dönüştür
                        df = pd.DataFrame(json_verisi)
                        # CSV dosyasına kaydet
                        df.to_csv("veri.csv", index=False)        
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
                        if df.empty:
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
                        delete_columns = []
                        missing_columns = []
                        op = ["Remove Duplicate Rows", #Yinelenen satırları kaldırma 
                                      "Lowercase Column Names",  #Sütun isimlerini küçük harfe çevirme
                                      "Lowercase Values in Object Data," #Kategorik verileri küçük harfe çevirme
                                      "Delete Unnecessary Columns", #Gereksiz kolonları silme
                                      "Fill Missing Values with -Unknown- in String Values", 
                                      "Label Encoding", 
                                      "One-Hot Encoding", 
                                      "Fill Missing Values in Selected Columns"]
                        
                        operations = st.multiselect("Choose cleaning operations:", op)
                        if "Delete Unnecessary Columns" in operations:
                                delete_columns = st.multiselect("Silinmesini istediğiniz sütunları seçin: ", df.columns.tolist())
                        if "Find Columns with Missing Values" in operations:
                                missing_columns = st.multiselect("Eksik değerleri doldurmak istediğiniz kolon isimleri", df.columns.tolist())

                        clean_btn = st.button("Clean the Dataset", type="primary", use_container_width=True) 
                        if clean_btn:
                                prepare_data(df, operations, delete_columns, missing_columns)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
                st.error('Yüklenen veri seti boş veya geçersiz.', icon="🚨")
        except Exception as e:
                st.error(f'Bir hata oluştu: {e}', icon="🚨")
                        



        