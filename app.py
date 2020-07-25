#importing libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_ngrok import run_with_ngrok
import pickle
from skimage.io import imread
from skimage.transform import resize
from numpy import load,array,argmax
from tensorflow.keras.models import model_from_json
import urllib.request
import os
import requests
from flask import Flask, render_template, session, redirect, url_for, session
import requests
import base64
import sys



def decode(base64_string):
    if isinstance(base64_string, bytes):
        base64_string = base64_string.decode("utf-8")
    imgdata = base64.b64decode(base64_string)
    img = imread(imgdata, plugin='imageio')
    return img

#taken from this StackOverflow answer: https://stackoverflow.com/a/39225039



app = Flask(__name__)
run_with_ngrok(app)#loading the model weights


with open("model_num.json", "r") as json_file:
    model = model_from_json(json_file.read())

model.load_weights('modelo1_weights.h5')


# model = load_model(destination)
@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        message = request.form['message']
        data = message
        # generator = single_picture_loader(data)
        image = decode(data)
        image = resize(image, (224, 224,3))
        image = [image]
        image = np.array(image)/255.0
        image = [image for _ in range(3)]
        pred = model.predict(image, batch_size=1)
        pred = np.argmax(pred, axis = 1)
        # #como yo lo plantee
        # # 0 neumonía
        # # 1 covid
        # # 2 sano
        # resultado = 0
        if pred[0] == 0:
            resultado = 1
        if pred[0] == 1:
            resultado = 2
        if pred[0] == 2:
            resultado = 0
        print(resultado,"CLASSIFICATION")
        
    
    return render_template('index2.html', generator_data=generator , value=message)
if __name__ == "__main__":
    
    app.run()