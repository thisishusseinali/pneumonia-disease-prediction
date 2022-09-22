import streamlit as st
import numpy as np
import predictor
from PIL import Image
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


st.image('header.png')
# Introduction text
st.markdown(unsafe_allow_html=True, body="<h4>Welcome to pneumonia detection app </h4>"
                                         "<p>This is a basic app built with Streamlit."
                                         "With this app, you can upload a Chest X-Ray image and predict if the patient "
                                         "from that image suffers pneumonia or not.</p>"
                                         "<p>The model used is a Convolutional Neural Network (CNN) and in this "
                                         "moment has a test accuracy of "
                                         "<strong>90.7%.</strong></p>")
st.markdown("First, let's load an X-Ray Chest image.")

imageLocation = st.empty()
imageLocation.image('image.jpg')
img = st.file_uploader(label="Load X-Ray Chest image", type=['jpeg', 'jpg', 'png'], key="xray")

if img is not None:
    # Preprocessing Image
    p_img = predictor.preprocess_image(img)
    imageLocation.image(img)

    # Loading model
    loading_msg = st.empty()
    loading_msg.text("Predicting...")
    model = predictor.load_model()

    # Predicting result
    prob, prediction = predictor.predict(model, p_img)

    loading_msg.text('')

    if prediction:
        st.markdown(unsafe_allow_html=True, body="<span style='color:red; font-size: 50px'><strong><h4>Pneumonia! :slightly_frowning_face:</h4></strong></span>")
    else:
        st.markdown(unsafe_allow_html=True, body="<span style='color:green; font-size: 50px'><strong><h3>Healthy! :smile: </h3></strong></span>")

    st.text(f"*Probability of pneumonia is {round(prob[0][0] * 100, 2)}%")
st.image('footer.png')
