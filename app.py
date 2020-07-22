
file_id = '1fVmpa2omDEjlm5PZkxHBhYmgnMpkHglq'
destination = 'model.h5'
download_file_from_google_drive(file_id, destination)
#importing libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_ngrok import run_with_ngrok
import pickle#creating the flask object
app = Flask(__name__)
run_with_ngrok(app)#loading the model weights

model = load_model(destination)
@app.route('/')
def home():
    return render_template('index.html')@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = model.predict(vect)
        if my_prediction == 1:
            output = "a Spam"
        elif my_prediction == 0:
            output = "Not a Spam"
    
    outputs = 'This email is '+output
    return render_template('index2.html', prediction_text=outputs , value=message)
if __name__ == "__main__":
    app.run()