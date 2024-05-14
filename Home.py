import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_card import card
from streamlit_lottie import st_lottie  # pip install streamlit-lottie
import json
import pandas as pd
import requests
from PIL import Image
from st_social_media_links import SocialMediaIcons


# Page configuration
st.set_page_config(
    page_title="ATOM AI",
    layout="wide",
)

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

# Title and Introduction
st.title("*Welcome to* Atom AI")
st.write(
    "As Atom AI family, we are proud to introduce you our new project. In this project, we enable you to use artificial intelligence in an easy way by performing data set cleaning, automatic machine learning and deep learning operations."
)

# Function to load Lottie animations (replace with your filepaths)
def load_lottiefile(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding = load_lottiefile("media/lottieFiles/visionMission.json")  # replace link to local lottie file
lottie_hello = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_M9p23l.json")

lottie_ai = load_lottiefile("media/lottieFiles/ai.json")  # replace link to local lottie file
lottie_comp = load_lottiefile("media/lottieFiles/computer.json")  # replace link to local lottie file


def btn_click(choice):
    if choice == "Yes":
        print("Yes butonuna tıklandı.")
        st.markdown(f"[Datasets Sayfasına Git]({'http://localhost:8501/Datasets'})")
    elif choice == "No":
        print("No butonuna tıklandı.")
        
with st.container(border=True):
    st.write("Have you organized your data set?")
    col1, col2 = st.columns(2)
    with col1:
        yes_btn = st.button(label="Yes, Let's Train & Test!", key="yes_btn", use_container_width=True)

    with col2:
        no_btn = st.button(label="No, Let's Do It!", use_container_width=True)

if yes_btn:
    btn_click("Yes")
elif no_btn:
    btn_click("No")


# Tabs with content
tab1, tab2, tab3, tab4 = st.tabs(
    ["Vision & Mission", "Why AutoML?", "What are we doing?", "What are we aiming for?"]
)

with tab1:
    col1, col2 = st.columns([1, 3])
    with col1:
        st_lottie(
            lottie_coding,
            speed=1,
            reverse=False,
            loop=True,
            quality="low",  # medium ; high
            width=None,
            key=None,
            height=250,
        )

    with col2:
        st.subheader("Our Vision")
        st.caption(
            "Its vision is to enable users who do not have a basic knowledge of data science, machine learning and deep learning to gain the ability to perform effective analysis on their own data sets, automate data cleaning and pre-processing steps, evaluate different machine learning models and select the most appropriate model, and make real-time predictions."
        )
        st.subheader("Our Mission")
        st.caption(
            "Focusing on the needs of individuals who do not have basic knowledge in the fields of data science, deep learning and machine learning, we aim to provide an interactive learning and application platform that will enable them to improve their skills in these areas."
        )

# Content for other tabs (fill in details)
with tab2:
    st.subheader("Why AutoML?")
    st.caption("The project aims to enable users who do not have even a basic knowledge of data science, machine learning and deep learning to gain the ability to perform effective analysis on their own data sets, automate data cleaning and pre-processing steps, evaluate different machine learning models and select the best model, and make real-time predictions. Data science and machine learning are increasingly in demand among professionals in different sectors, allowing individuals with these skills to gain a competitive advantage in the business world. The aim of this project is to focus on the needs of individuals who have even a basic knowledge of data science, deep learning and machine learning and who want to operate in this field, and to provide an interactive learning and application platform designed to improve their skills in this field. Another achievement of the project is that it enables the selection of the most appropriate model, the selection of the best parameters with Hyperparameter Optimization and fast data analysis without the need to waste time writing code for machine learning and deep learning from scratch with data.")

with tab3:
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.subheader("What Are We Doing?")
        st.caption("The automatic c machine learning (Auto ML) approach over 19 different omatic machine learning (Auto ML) approach over 19 different omatic machine learning (Auto ML) approach over 19 different pproach over 19 different models to be used in the project increases the flexibility and wide applicability of the project. This project, which will be written in Python, is part of an emerging popular area to meet the requirements in data science, deep learning and machine learning. With 19 different models, it provides users with the ability to choose any model they want and find most of the models they are looking for, as well as machine learning that automates the training, tuning and deployment of machine learning models.")
    with col2: 
        st_lottie(
            lottie_comp,
            speed=1,
            reverse=False,
            loop=True,
            quality="low",  # medium ; high
            width=None,
            key=None,
            height=200,
        )

with tab4:
    st.subheader("What Are We Aiming For?")
    st.caption("•Data Cleaning and Preprocessing**: The project automates data cleaning steps such as filling in missing data, deleting unnecessary columns. It also allows users to perform basic pre-processing steps such as Label Encoding or One-Hot Encoding. ")
    st.caption("•Model Selection and Training**: By presenting multiple machine learning models to users, it evaluates the performance of each with different parameters and gives them the chance to choose the best performing model.")
    st.caption("•Prediction Capability**: It allows users to make real-time forecasts for the data set at hand. The user can make predictions with the selected model based on the data entered manually.")

with st.container():
    col1, col2, col3 = st.columns(3)
    col1.metric("Temperature", "70 °F", "1.2 °F")
    col2.metric("Wind", "9 mph", "-8%")
    col3.metric("Humidity", "86%", "4%")

# Introduction section with video and image
st.subheader("Introduction")
st.video("media/videos/software.mp4")

with st.container(border=True):
    st.subheader("This is inside the container")
    col1, col2 = st.columns([0.7, 0.3])
    # İlk sütunda metin
    with col1:
        st.write("Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy Lorem Ipsum is simply dummy text of the printing and typesetting industry. LoLorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy rem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.")
    # İkinci sütunda görüntü
    with col2:
        st.image("media/images/img.png", caption="We are study for future.", width=300)


st.subheader("Yapay Zeka hakkında daha fazlasını öğren.")
st.write("Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.")

col1, col2, col3 = st.columns(3)
with col1:
        hasClicked = card(
            title="Hello World!",
            text="Some description",
            image="http://placekitten.com/200/300",
            url="https://medium.com/@raja.gupta20/generative-ai-for-beginners-part-1-introduction-to-ai-eadb5a71f07d",
            styles={
            "card": {
                "width": "300px",
                "height": "200px",
                "box-shadow": "0 0 10px rgba(0,0,0,0.5)",
                
            "text": {
                "font-family": "serif",
                
            }
        }
    }
    )

with col2:
        hasCld = card(
            title="sdsdd!",
            text="Somedssdption",
            #image="media/images/img.png",
            url="https://medium.com/tag/artificial-intelligence",
            styles={
            "card": {
                "width": "300px",
                "height": "200px",
                "box-shadow": "0 0 10px rgba(0,0,0,0.5)",
                
            "text": {
                "font-family": "serif",
                
                }
            } 
        }
    )

with col3:
        res = card(
        title="Streamlit Card",
        text="This is a test card",
        #image="https://placekitten.com/500/500",
        url="https://odsc.medium.com/10-best-books-to-teach-you-about-artificial-intelligence-in-2024-0cbd07a45437",
        styles={
            "card": {
                "width": "300px",
                "height": "200px",
                "box-shadow": "0 0 10px rgba(0,0,0,0.5)",
                
            "text": {
                "font-family": "serif",   
            }
        } 
    }
    )

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
    st.write("Lorem Ipsum has been the industry's standard dummy Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy ")

st.info('If you want to know more in detail, read our manual.', icon="ℹ️")



with st.container(border=True):
    col1, col2, col3 = st.columns([0.4, 0.3, 0.3])
    with col1:
        st.caption("Dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy")
    with col2:
            st.image("media/images/mustafa_can.png", width=200)
            st.write("Mustafa Can")
            insta, link, mail = st.columns(3)
            with col1:
                 st.write("LinkedIn")
                 st.markdown('<a href="https://www.linkedin.com/search/results/people/?firstName=Mustafa&keywords=must&lastName=Mustafa&origin=GLOBAL_SEARCH_HEADER&sid=C~F" target="_blank"><img src="/media/images/Lin.png" width="50" height="50"></a>', unsafe_allow_html=True)
            
    with col3:
            st.image("media/images/tugce_ulucan.png", width=200 )
            st.write("Tuğçe Ulucan")
            im = Image.open("C:/Users/ACER/Documents/GitHub/AutoMl-Project/media/images/li.png")
            """
            [![Follow](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](mailto:fatmatugcelcn@gmail.com)
            [![Follow](https://img.shields.io/badge/instagram-1DA1F2?style=for-the-badge&logo=instagram&logoColor=white)](https://www.linkedin.com/in/fatmatugceulucan/)

            """
            ig = st.btn()


with st.container(border=True):
     
     col1, col2 = st.columns(2)
     with col1:
          form = st.form("my_form")
          form.subheader("Contact With Us")
          form.subheader("*Authors*")
          form.write("Mustafa Can ")
          st.caption("0541 342 7890")
          st.caption("mustafa@gmail.com")
          st.write("Fatma Tuğçe Ulucan ")
          st.caption("0552 629 0420")
          st.caption("fatmatugcelcn@gmail.com")

          with st.form("my_form"):
            st.write("Inside the form")
            my_number = st.slider('Pick a number', 1, 10)
            my_color = st.selectbox('Pick a color', ['red','orange','green','blue','violet'])
            st.form_submit_button('Submit my picks')

            # This is outside the form
            st.write(my_number)
            st.write(my_color)

     with col2:
        data = {
        'City': ['Istanbul'],
        'Latitude': [41.0082],  # İstanbul'un enlem değeri
        'Longitude': [28.9784]  # İstanbul'un boylam değeri
        }
        df = pd.DataFrame(data)

        # Haritayı gösterme
        st.map(df, latitude='Latitude', longitude='Longitude')
    

