import streamlit as st
import PIL
import numpy as np
from tensorflow import *
import time
import os
from PIL import Image
import matplotlib.image as mpimg
from fastai.vision import *
from PIL import Image, ImageOps
###############################################################################################
def pneumonia_predictor(file_path):
    #tf.image.resize(file_path, size=(32,32))
    #pavan_img    = image.resize(file_path,size=(500, 500))
    pavan_img    = image.load_img(file_path, target_size=(500, 500),color_mode='grayscale')
    pp_pavan_img = image.img_to_array(pavan_img)
    pp_pavan_img = pp_pavan_img/255
    pp_pavan_img = np.expand_dims(pp_pavan_img, axis=0)
    pneumonia_model = load_model("models/pneu_cnn_model.h5")
    
    pavan_preds= pneumonia_model.predict(pp_pavan_img)
    if pavan_preds>= 0.5:
        label,accuracy = 'Pneumonia',pavan_preds[0][0]
        return  label,accuracy
    else:
        label,accuracy = 'Normal',1-pavan_preds[0][0]
        return  label,accuracy
###############################################################################################
st.title("Pneumonia Test Application")
st.header("Classification Example")

option = st.radio('', ['Choose a Sample XRay', 'Upload your own XRay'])

if option == 'Choose a Sample XRay':
    # Get a list of test images in the folder
    test_imgs = os.listdir("uploads/")
    test_img = st.selectbox('Please Select a Test Image:',test_imgs)
    # Display and then predict on that image
    fl_path = ("uploads/"+test_img)
    img = open_image(fl_path)

    display_img = mpimg.imread(fl_path)
    st.image(display_img, caption="Chosen XRAY", use_column_width=True)
    
    st.write("")
    with st.spinner("Identifying the Disease..."):
        time.sleep(5)
    label, prob = pneumonia_predictor(img)
    st.success(f"Image Disease: {label}, Confidence: {prob:.2f}%")

elif option == 'Upload your own XRay':
    uploaded_file = st.file_uploader("Choose an Image", type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded XRAY", use_column_width=True)
        
        img = image.pil2tensor(img, np.float32).div_(255)
        img = image.Image(img)
        st.write("")
        
        with st.spinner("Identifying the Disease..."):
            time.sleep(5)
        label, prob = pneumonia_predictor(img)
        st.success(f"Image Disease: {label}, Confidence: {prob:.2f}%")
###############################################################################################
st.warning("NOTE: IF YOU UPLOAD AN IMAGE WHICH IS NOT A CHEST XRAY, THE MODEL WILL GIVE VERY WIERD PREDICTIONS BECAUSE IT'S TRAINED TO IDENTIFY WHICH ONE OF THE 2 LABELS THE MODEL IS MOST CONFIDENT OF.")
st.write("PROJECT MADE WITH ❤️ BY : HUSSEIN ALI")
