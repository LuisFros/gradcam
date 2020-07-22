#importing libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_ngrok import run_with_ngrok
import pickle#creating the flask object
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
from skimage.io import imread
from skimage.transform import resize
from numpy import load,array,argmax
import numpy as np
import urllib.request
import os
import requests
from flask import Flask, render_template, session, redirect, url_for, session
import requests
import base64


def decode(base64_string):
    if isinstance(base64_string, bytes):
        base64_string = base64_string.decode("utf-8")

    imgdata = base64.b64decode(base64_string)
    img = imread(imgdata, plugin='imageio')
    return img

class My_Custom_Generator(Sequence) :
  
  def __init__(self, image_filenames, labels, batch_size,augmented=False) :
    self.image_filenames = image_filenames
    self.labels = labels
    self.batch_size = batch_size
    self.augmented = augmented
    
  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

  def __getitem__(self, idx) :
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

    images = [decode(file_name) for file_name in batch_x]
    label_list = np.array(batch_y)
    
    imag = np.array([resize(img,(256, 256,3)) for img in images])/255.0
    return imag,label_list

#taken from this StackOverflow answer: https://stackoverflow.com/a/39225039

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def single_picture_loader(img_path):
  BATCH_SIZE = 1
  UNUSED_LABEL = [1]
  IMAGES = [img_path]
  return My_Custom_Generator(IMAGES, UNUSED_LABEL,BATCH_SIZE)


app = Flask(__name__)
run_with_ngrok(app)#loading the model weights
file_id = '1fVmpa2omDEjlm5PZkxHBhYmgnMpkHglq'
destination = 'model.h5'
download_file_from_google_drive(file_id, destination)
model = load_model(destination)
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
        generator = single_picture_loader(data)
        my_prediction = model.predict(generator)
        if my_prediction == 1:
            output = "a Spam"
        elif my_prediction == 0:
            output = "Not a Spam"
    
    outputs = 'This email is '+output
    return render_template('index2.html', prediction_text=outputs , value=message)
if __name__ == "__main__":
    
    app.run()