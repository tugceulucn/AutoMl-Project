import streamlit as st
import os, json, requests, pandas as pd

from PIL import Image
from streamlit_card import card
from streamlit_lottie import st_lottie
from annotated_text import annotated_text
from streamlit_option_menu import option_menu
from st_social_media_links import SocialMediaIcons

# Lottie animasyonlarını yükleme işlevi (dosya yollarınızla değiştirin)
def load_lottiefile(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return str(e)
    
def frontend():
    # Başlık ve giriş
    st.title("*Welcome to* Atom AI")
    st.write(
        "As Atom AI family, we are proud to introduce you our new project. In this project, we enable you to use artificial intelligence in an easy way by performing automatic data set cleaning-analysis, automatic machine learning and deep learning operations.")

    #Kullanılan animasyonların tanımlanması
    lottie_coding = load_lottiefile("media/lottieFiles/visionMission.json")  # replace link to local lottie file
    lottie_ai = load_lottiefile("media/lottieFiles/ai.json")  # replace link to local lottie file
    lottie_comp = load_lottiefile("media/lottieFiles/computer.json")  # replace link to local lottie file

    st.info("If you want to edit your dataset, you should go to the **📈 Datasets** page. If your data set is organized, you can move on to the **🌍 Machine Learning** or **📊 Deep Learning** pages.", icon="ℹ️")
        
    # Uygulama hakkında bilgi veren "About" sekmeleri
    about1, about2, about3 = st.tabs(["Vision & Mission", "What are we doing?", "What are we aiming for?"])
    with about1:
        animation, text = st.columns([1, 3]) #Sekme kolonlarında 1'e 3 oran vardır.
        with animation:
            st_lottie(lottie_coding, speed=1,
                reverse=False, loop=True,
                quality="low",  # medium ; high 
                width=None, key=None, height=250)
        with text:
            st.subheader("Our Vision")
            st.write("Its vision is to enable users without basic knowledge in data science, machine learning and deep learning to perform effective analysis on their own data sets, automate data cleaning and pre-processing steps, evaluate different machine learning and deep learning models, select the most appropriate model and make real-time predictions.")
            st.subheader("Our Mission")
            st.write("Mission is to provide an interactive automated artificial intelligence platform by focusing on the problems and needs in the fields of data science and artificial intelligence.")
    with about2:
        text, animation = st.columns([0.7, 0.3])
        with text:
            st.subheader("What Are We Doing?")
            st.write("Autom AI allows users to analyze data sets, edit data sets, perform machine learning and deep learning without the need for software. Machine learning is a more stable and easier version of deep learning. Autom AI only supports CSV and Json data sets for machine learning. It supports 20 different machine learning models. Deep learning is more customizable than machine learning. Autom AI only supports the Image data type for deep learning. it is useful because it supports a variety of features.")
        with animation: 
            st_lottie(lottie_comp, speed=1,
                reverse=False, loop=True,
                quality="low",  # medium ; high
                width=None, key=None, height=200)
    with about3:
        st.subheader("What Are We Aiming For?")
        st.write("We want to enable people with or without software knowledge to develop their own artificial intelligence projects without using any programming language, using their own data sets or using various ready-made data sets in the application to perform automatic data analysis and automatic data editing, and we want to enable them to develop, record and test their projects by finding the best features of their projects using a helper language model together with more specialized models.")

    # Metinleri tanımlama
    pages_info = {
        "Home": "It is the first page encountered when entering the website. Home page is the page that has information about the vision and mission of the application, information and directions to introduce the application such as what it does.",
        "Datasets": "This page shows the datasets uploaded by the user. Data editing, label encoding, one hot encoding, standardization and normalization operations are performed on this page.",
        "Machine Learning": "Here, the user can train the selected model on the uploaded data sets. If they wish, they can download the trained model in the format they want and predict the target by entering sample data.",
        "Deep Learning": "They can train a deep learning model with the selected features over the data set uploaded here. If they wish, they can download the trained model in the format they want and predict the target by entering sample data."
    }

    # DataFrame oluşturma
    df = pd.DataFrame(list(pages_info.items()), columns=["Page", "Description"])
    st.write("**OUR PAGES**", df)
    # Bilgilendirici İstatistklerin verilmesi
    with st.container():
        col1, col2, col3 = st.columns(3)
        col1.metric("Number of people to work on AI", "97 Million", "By 2025")
        col2.metric("Artificial intelligence industry value", "More Than 13 Times.", "Next 7 Years ")
        col3.metric("The global AI market is worth over", "$196 billion", "Increase Quickly")

    # Tanıtım Videosu Gösterilmesi
    st.subheader("Introduction")
    st.video("media/videos/software.mp4")

    #Bilgilendirme: Makine öğrenmesi ve derin öğrenme nedir.
    col1, col2 = st.columns([0.7,0.3])
    with col1:
        st.write("**Machine Learning & Deep Learning**")
        st.write("Machine learning is a branch of artificial intelligence that enables computers to learn by analyzing data. These algorithms extract patterns from data and make predictions. It is divided into three main categories: supervised, unsupervised, and reinforcement learning. Supervised learning learns the relationship between input and output. Unsupervised learning discovers patterns in the data. Reinforcement learning tries to optimize behavior while receiving feedback. Machine learning is used in various fields such as image and speech recognition, natural language processing, medical diagnosis, and autonomous driving.")
    with col2:
            st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQET6D5yHvhXjGU21lrG0L35CgQpJH93KQ-MXUG7AHa1g&s", caption="We are study for future.")
    
    st.write("Deep learning is the process of learning from complex data using multi-layered artificial neural networks. Algorithms learn features and patterns in the dataset. It excels in large datasets and complex tasks, such as image and speech recognition, natural language processing, and game playing. It offers a structure that enables deeper and more detailed analysis by improving inter-layer learning and data representation.")
    #Kullanıcıyı daha fazla bilgi için örnek linklere (card) yönlendirme
    st.subheader("Learn More More More")
    st.write("There are also various articles and researches written in the field of artificial intelligence that we utilized in the development of the Atom AI project. These articles cover different application areas of AI technologies, innovative methods and the latest developments. In order to increase the knowledge in this field and to access the right information, it is of great importance to read the articles from reliable and academic sources. We wanted to share a few resources for you. As Atom AI developers, we have always made it a principle to benefit from accurate and reliable sources. **Stay tuned for more!** ")
   
    #Örnek eğitici kaynaklar paylaşılması.
    url1, url2, url3 = st.columns(3)
    with url1:
            card1 = card(
                title="What is Artificial Intelligence?",
                text="How does AI work and future of it.",
                image= "https://miro.medium.com/v2/resize:fit:1400/format:webp/1*ac3i0wBER2lBquBaOGEcsQ.jpeg",
                url="https://medium.com/@mygreatlearning/what-is-artificial-intelligence-how-does-ai-work-and-future-of-it-d6b113fce9be",
                styles={
                "card": {
                    "width": "300px",
                    "height": "200px",
                    "box-shadow": "0 0 10px rgba(0,0,0,0.5)",
                "text": {"font-family": "serif",}
                        }
                    }
                )
    with url2:
            card2 = card(
                title="Artificial Intelligence",
                text="More content on Medium",
                image="https://miro.medium.com/v2/resize:fit:2000/format:webp/0*8pH57xqqk484OljX",
                url="https://medium.com/tag/artificial-intelligence",
                styles={
                "card": {
                    "width": "300px",
                    "height": "200px",
                    "box-shadow": "0 0 10px rgba(0,0,0,0.5)",    
                "text": {"font-family": "serif"}
                    } 
                })
    with url3:
            card3 = card(
            title="Deep Learning",
            text="Understanding Basic Neural Networks",
            image="https://miro.medium.com/v2/resize:fit:1400/format:webp/0*KyqJ3ywjYSBBM5RS.jpg",
            url="https://medium.com/@Coursesteach/deep-learning-part-1-86757cf5a0c3",
            styles={
                "card": {
                    "width": "300px",
                    "height": "200px",
                    "box-shadow": "0 0 10px rgba(0,0,0,0.5)",  
                "text": {"font-family": "serif"}
                } 
            })

    #Bilgilendirme: Derin öğrenme ve yapay zeka farkı
    col1, col2 = st.columns([0.3, 0.7])
    with col1: 
        st_lottie(
                lottie_ai,
                speed=1,
                reverse=False,
                loop=True,
                quality="low",  # medium ; high
                width=None,
                key=None,
                height=250,
            ) 
    with col2:
        st.write("Machine Learning (ML) is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed, and it is divided into three main categories: supervised, unsupervised, and reinforcement learning. ML algorithms discover patterns and relationships in data and use this information for future predictions or decisions. Deep Learning (DL), on the other hand, is a subset of ML that learns from more complex data using multi-layered artificial neural networks. DL excels particularly in large datasets and complex tasks, such as image and speech recognition. Deep Learning offers a sophisticated and effective approach to solving more intricate problems compared to Machine Learning by improving inter-layer learning and data representation, enabling deeper and more detailed analysis.")

    directory = "media/texts"
    
    if not os.path.isdir(directory):
        st.error(f"Klasör mevcut değil: {directory}")
        return

    file_list = [f for f in os.listdir(directory) if f.endswith(".txt")]
    
    if not file_list:
        st.error("Klasörde .txt dosyası bulunamadı.")
        return

    selected_file = st.selectbox("Hangi dosyayı görüntülemek istiyorsun?", file_list)

    if selected_file:
        file_path = os.path.join(directory, selected_file)
        content = read_text_file(file_path)
        #st.write(f"Dosya: {selected_file}")
        st.code(content, language="Python")
    
    st.info('If you want to know more in detail, read our [**manual**](https://www.retmon.com/blog/veri-gorsellestirme-nedir#:~:text=Veri%20g%C3%B6rselle%C5%9Ftirme%3B%20verileri%20insan%20beyninin,i%C3%A7in%20kullan%C4%B1lan%20teknikleri%20ifade%20eder.).', icon="ℹ️")

    #Ekibi tanıtma.
    with st.container(border=True):
        col1, col2= st.columns([0.7, 0.3])
        with col1:
                st.write("Atom AI project was developed as a graduation project of Istinye University Software Engineering department. The project developers are Mustafa Can and Fatma Tuğçe Ulucan. Throughout their university years, the developers have worked together on various projects in the field of artificial intelligence, continuously increasing their knowledge and experience in this field. Their cooperation and harmony in joint projects formed the basis of the Atom AI project. Atom AI is a project focused on automating artificial intelligence that automates everything. This project aims to use artificial intelligence more efficiently and effectively in various fields. In the future, Mustafa Can and Fatma Tuğçe Ulucan plan to continue their work on automating artificial intelligence by further developing the Atom AI project. These studies will enable a wider range of artificial intelligence technologies to be used and will pioneer innovative solutions in this field.")
                c1, c2, c3, c4 = st.columns([0.2,0.3,0.2,0.3])
                with c1:
                    st.write("Mustafa CAN")
                with c2:
                    """[![Follow](https://img.shields.io/badge/linkedin-1DA1F2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mustafa-can369/)"""
                with c3:
                    st.write("Fatma Tuğçe Ulucan")
                with c4:
                    """[![Follow](https://img.shields.io/badge/linkedin-1DA1F2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/fatmatugceulucan/)"""
        with col2:
                st.image("media/images/devolopers.png", width=300 )

    st.image("media/images/footer.png",use_column_width=True)

if __name__ == "__main__":
     # Page configurations
    st.set_page_config(page_title="ATOM AI", layout="wide",)
    hide_default_format = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """
    st.markdown(hide_default_format, unsafe_allow_html=True)

    frontend()