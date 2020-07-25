#importing libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_ngrok import run_with_ngrok
import pickle
from skimage.io import imread
from skimage.transform import resize
from numpy import load,array,argmax
import urllib.request
import os
import requests
from flask import Flask, render_template, session, redirect, url_for, session
import requests
import base64
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Input, Dense, Flatten, AveragePooling2D, Dropout, BatchNormalization, Conv2D, concatenate
import tensorflow
import cv2
import sys

INP_SIZE = (224,224,3)
def generate_DenseNet_model():
    model = DenseNet121(
        include_top = False,
        weights = 'imagenet',
        input_tensor = Input(shape=INP_SIZE),
    )
    return model

def define_stacked_model(members):
	for i in range(len(members)):
		model = members[i]
		for layer in model.layers:
			layer.trainable = False
			layer._name = 'ensemble_' + str(i+1) + '_' + layer.name
	ensemble_visible = [model.input for model in members]
	ensemble_outputs = [model.output for model in members]
	merge = concatenate(ensemble_outputs)
	headModel = AveragePooling2D(pool_size=(4, 4))(merge)
	headModel = Flatten(name="flatten")(headModel)
	headModel = Dense(100, activation="relu")(headModel)
	headModel = Dropout(0.5)(headModel)
	output = Dense(3, activation="softmax")(headModel)
	model = Model(inputs=ensemble_visible, outputs=output)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


def generate_model():
    model = generate_DenseNet_model()
    model2 = tensorflow.keras.models.clone_model(model)
    model3 = tensorflow.keras.models.clone_model(model)
    members = [model, model2, model3]
    model = define_stacked_model(members)
    return model



def decode(base64_string):
    if isinstance(base64_string, bytes):
        base64_string = base64_string.decode("utf-8")

    imgdata = base64.b64decode(base64_string)
    img = imread(imgdata, plugin='imageio')
    return img

#taken from this StackOverflow answer: https://stackoverflow.com/a/39225039



app = Flask(__name__)
run_with_ngrok(app)#loading the model weights

model = generate_model()
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
        data = [message]
        # generator = single_picture_loader(data)
        print(model.summary())
        image = decode(data)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
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