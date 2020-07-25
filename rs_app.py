
import pickle
from skimage.io import imread
from skimage.transform import resize
from numpy import array,argmax,uint8
from tensorflow.keras.models import model_from_json,load_model
import urllib.request
import os
import requests
from flask import Flask, render_template, session, redirect, url_for, session
import requests
import base64
import sys
import streamlit as st
from PIL import Image, ImageOps

def decode(base64_string):
    if isinstance(base64_string, bytes):
        base64_string = base64_string.decode("utf-8")
    imgdata = base64.b64decode(base64_string)
    img = imread(imgdata, plugin='imageio')
    return img


def import_and_predict(data, model):
    
        image = decode(data)
        image = resize(image, (224, 224,3))
        image = [image]
        image = array(image,dtype=uint8)/255.0
        image = [image for _ in range(3)]
        pred = model.predict(image, batch_size=1)
        
        return pred

model = load_model('my_model.hdf5')

st.write("""
         # Rock-Paper-Scissor Hand Sign Prediction
         """
         )

st.write("This is a simple image classification web app to predict rock-paper-scissor hand sign")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
#
if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("It is a paper!")
    elif np.argmax(prediction) == 1:
        st.write("It is a rock!")
    else:
        st.write("It is a scissor!")
    
    st.text("Probability (0: Paper, 1: Rock, 2: Scissor)")
    st.write(prediction)