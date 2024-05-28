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
        
def frontend():
    # Başlık ve giriş
    st.title("*Welcome to* Atom AI")
    st.write(
        "As Atom AI family, we are proud to introduce you our new project. In this project, we enable you to use artificial intelligence in an easy way by performing data set cleaning, automatic machine learning and deep learning operations.")

    #Kullanılan animasyonların tanımlanması
    lottie_coding = load_lottiefile("media/lottieFiles/visionMission.json")  # replace link to local lottie file
    lottie_ai = load_lottiefile("media/lottieFiles/ai.json")  # replace link to local lottie file
    lottie_comp = load_lottiefile("media/lottieFiles/computer.json")  # replace link to local lottie file

    st.info("If you want to edit your dataset, you should go to the **📈 Datasets** page. If your data set is organized, you can move on to the **🌍 Machine Learning** or **📊 Deep Learning** pages.", icon="ℹ️")
        
    # Uygulama hakkında bilgi veren "About" sekmeleri
    about1, about2, about3, about4 = st.tabs(["Vision & Mission", "Why AutoML?", "What are we doing?", "What are we aiming for?"])
    with about1:
        animation, text = st.columns([1, 3]) #Sekme kolonlarında 1'e 3 oran vardır.
        with animation:
            st_lottie(lottie_coding, speed=1,
                reverse=False, loop=True,
                quality="low",  # medium ; high 
                width=None, key=None, height=250)
        with text:
            st.subheader("Our Vision")
            st.caption("Its vision is to enable users who do not have a basic knowledge of data science, machine learning and deep learning to gain the ability to perform effective analysis on their own data sets, automate data cleaning and pre-processing steps, evaluate different machine learning models and select the most appropriate model, and make real-time predictions.")
            st.subheader("Our Mission")
            st.caption("Focusing on the needs of individuals who do not have basic knowledge in the fields of data science, deep learning and machine learning, we aim to provide an interactive learning and application platform that will enable them to improve their skills in these areas.")
    with about2:
        st.subheader("Why AutoML?")
        st.caption("The automated machine learning (AutoML) approach leverages more than 19 different models, which gives our project flexibility and wide applicability. Written in Python, this project is situated in a popular and emerging field that meets the requirements in data science, deep learning and machine learning. AutoML automates the training, tuning and deployment of machine learning models, allowing users to select and use the most appropriate model. This reduces the complexity of data science and artificial intelligence applications, reaching a wider audience of users. Data science and machine learning are increasingly in demand among professionals across different industries, enabling individuals with these skills to gain a competitive advantage in the business world. Traditional machine learning processes are often complex and time-consuming. Steps such as code writing, model selection, hyperparameter optimization and data preprocessing require technical knowledge and experience. However, AutoML automates these processes, enabling users to achieve faster and more effective results.")
    with about3:
        text, animation = st.columns([0.7, 0.3])
        with text:
            st.subheader("What Are We Doing?")
            st.caption("The automatic c machine learning (Auto ML) approach over 19 different omatic machine learning (Auto ML) approach over 19 different omatic machine learning (Auto ML) approach over 19 different pproach over 19 different models to be used in the project increases the flexibility and wide applicability of the project. This project, which will be written in Python, is part of an emerging popular area to meet the requirements in data science, deep learning and machine learning. With 19 different models, it provides users with the ability to choose any model they want and find most of the models they are looking for, as well as machine learning that automates the training, tuning and deployment of machine learning models.")
        with animation: 
            st_lottie(lottie_comp, speed=1,
                reverse=False, loop=True,
                quality="low",  # medium ; high
                width=None, key=None, height=200)
    with about4:
        st.subheader("What Are We Aiming For?")
        st.caption("• Data Cleaning and Preprocessing**: The project automates data cleaning steps such as filling in missing data, deleting unnecessary columns. It also allows users to perform basic pre-processing steps such as Label Encoding or One-Hot Encoding. ")
        st.caption("• Model Selection and Training**: By presenting multiple machine learning models to users, it evaluates the performance of each with different parameters and gives them the chance to choose the best performing model.")
        st.caption("• Prediction Capability**: It allows users to make real-time forecasts for the data set at hand. The user can make predictions with the selected model based on the data entered manually.")

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

    st.info('If you want to read our graduation thesis, you can access it from this [**link.**](https://www.youtube.com/watch?v=u1hSnYlEjn4)', icon="ℹ️")
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