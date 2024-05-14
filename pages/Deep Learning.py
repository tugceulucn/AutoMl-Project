import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_card import card
from streamlit_lottie import st_lottie  # pip install streamlit-lottie
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from bokeh.plotting import figure
import numpy as np

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

if selected == "Home":
        inputs, outputs = st.columns(2)

        with inputs:
                        tab1, tab2, tab3= st.tabs(["INPUTS", "EXAMPLE", "ABOUT"])
                        with tab1:
                                with st.container(border=True):
                                        st.write("INPUTS")
                                        st.caption("Şekillendirmek için veri kümeleri genellikle analizden önce ortadan kaldırılması gereken hatalar içerir. Bu hatalar, tahminleri önemli ölçüde etkileyebilecek yanlış yazılmış tarihler, parasal değerler ve diğer ölçüm birimleri gibi biçimlendirme hatalarını içerebilir. Aykırı değerler, sonuçları her durumda çarpıttığından önemli bir endişe kaynağıdır. Yaygın olarak bulunan diğer veri hataları; bozuk veri noktalarını, eksik bilgileri ve yazım hatalarını içerir. Temiz veriler, ML modellerinin yüksek oranda doğru olmasına yardımcı olabilir. Düşük kaliteli eğitim veri kümelerini kullanmak, dağıtılan modellerde hatalı tahminlere neden olabileceğinden temiz ve doğru veriler, makine öğrenimi modellerini eğitmek için özellikle önemlidir. Veri bilimcilerinin, zamanlarının büyük bir kısmını makine öğrenimi için veri hazırlamaya ayırmalarının başlıca nedeni budur.")
                                        with st.expander("🔗 Machine Learning"):
                                                st.caption("Bir şirket, karar vermeBir şirket, karar verme sürecini şekillendirmek içinBir şirket, karar verme sürecini şekillendirmek içinBir şirket, karar verme sürecini şekillendirmek içinBir şirket, karar verme sürecini şekillendirmek için sürecini şekillendirmek için verileri kullanırken alakalı, eksiksiz ve doğru verileri kullanmaları çok önemlidir. Bununla birlikte, veri kümeleri genellikle analizden önce ortadan kaldırılması gereken hatalar içerir. Bu hatalar, tahminleri önemli ölçüde etkileyebilecek yanlış yazılmış tarihler, parasal değerler ve diğer ölçüm birimleri gibi biçimlendirme hatalarını içerebilir. Aykırı değerler, sonuçları her durumda çarpıttığından önemli bir endişe kaynağıdır. Yaygın olarak bulunan diğer veri hataları; bozuk veri noktalarını, eksik bilgileri ve yazım hatalarını içerir. Temiz veriler, ML modellerinin yüksek oranda doğru olmasına yardımcı olabilir. Düşük kaliteli eğitim veri kümelerini kullanmak, dağıtılan modellerde hatalı tahminlere neden olabileceğinden temiz ve doğru veriler, makine öğrenimi modellerini eğitmek için özellikle önemlidir. Veri bilimcilerinin, zamanlarının büyük bir kısmını makine öğrenimi için veri hazırlamaya ayırmalarının başlıca nedeni budur.")
                                        with st.expander("🔗 Modeller"):
                                                st.caption("Bir şirket, karar vermeBir şirket, karar verme sürecini şekillendirmek içinBir şirket, karar verme sürecini şekillendirmek içinBir şirket, karar verme sürecini şekillendirmek içinBir şirket, karar verme sürecini şekillendirmek için sürecini şekillendirmek için verileri kullanırken alakalı, eksiksiz ve doğru verileri kullanmaları çok önemlidir. Bununla birlikte, veri kümeleri genellikle analizden önce ortadan kaldırılması gereken hatalar içerir. Bu hatalar, tahminleri önemli ölçüde etkileyebilecek yanlış yazılmış tarihler, parasal değerler ve diğer ölçüm birimleri gibi biçimlendirme hatalarını içerebilir. Aykırı değerler, sonuçları her durumda çarpıttığından önemli bir endişe kaynağıdır. Yaygın olarak bulunan diğer veri hataları; bozuk veri noktalarını, eksik bilgileri ve yazım hatalarını içerir. Temiz veriler, ML modellerinin yüksek oranda doğru olmasına yardımcı olabilir. Düşük kaliteli eğitim veri kümelerini kullanmak, dağıtılan modellerde hatalı tahminlere neden olabileceğinden temiz ve doğru veriler, makine öğrenimi modellerini eğitmek için özellikle önemlidir. Veri bilimcilerinin, zamanlarının büyük bir kısmını makine öğrenimi için veri hazırlamaya ayırmalarının başlıca nedeni budur.")

                        with tab2:
                                st.caption("Şekillendirmek için veri kümeleri genellikle analizden önce ortadan kaldırılması gereken hatalar içerir. Bu hatalar, tahminleri önemli ölçüde etkileyebilecek yanlış yazılmış tarihler, parasal değerler ve diğer ölçüm birimleri gibi biçimlendirme hatalarını içerebilir. Aykırı değerler, sonuçları her durumda çarpıttığından önemli bir endişe kaynağıdır. Yaygın olarak bulunan diğer veri hataları; bozuk veri noktalarını, eksik bilgileri ve yazım hatalarını içerir. Temiz veriler, ML modellerinin yüksek oranda doğru olmasına yardımcı olabilir. Düşük kaliteli eğitim veri kümelerini kullanmak, dağıtılan modellerde hatalı tahminlere neden olabileceğinden temiz ve doğru veriler, makine öğrenimi modellerini eğitmek için özellikle önemlidir. Veri bilimcilerinin, zamanlarının büyük bir kısmını makine öğrenimi için veri hazırlamaya ayırmalarının başlıca nedeni budur.")
                                with st.expander("🔗 Machine Learning"):
                                                st.caption("Bir şirket, karar vermeBir şirket, karar verme sürecini şekillendirmek içinBir şirket, karar verme sürecini şekillendirmek içinBir şirket, karar verme sürecini şekillendirmek içinBir şirket, karar verme sürecini şekillendirmek için sürecini şekillendirmek için verileri kullanırken alakalı, eksiksiz ve doğru verileri kullanmaları çok önemlidir. Bununla birlikte, veri kümeleri genellikle analizden önce ortadan kaldırılması gereken hatalar içerir. Bu hatalar, tahminleri önemli ölçüde etkileyebilecek yanlış yazılmış tarihler, parasal değerler ve diğer ölçüm birimleri gibi biçimlendirme hatalarını içerebilir. Aykırı değerler, sonuçları her durumda çarpıttığından önemli bir endişe kaynağıdır. Yaygın olarak bulunan diğer veri hataları; bozuk veri noktalarını, eksik bilgileri ve yazım hatalarını içerir. Temiz veriler, ML modellerinin yüksek oranda doğru olmasına yardımcı olabilir. Düşük kaliteli eğitim veri kümelerini kullanmak, dağıtılan modellerde hatalı tahminlere neden olabileceğinden temiz ve doğru veriler, makine öğrenimi modellerini eğitmek için özellikle önemlidir. Veri bilimcilerinin, zamanlarının büyük bir kısmını makine öğrenimi için veri hazırlamaya ayırmalarının başlıca nedeni budur.")


        with outputs:
                        with st.container(border=True):
                                st.write("OUTPUTS")
                        
else: 
        st.write("dsdsd")

        

