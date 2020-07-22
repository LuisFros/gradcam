from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file
from tensorflow.keras.utils import Sequence
# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.applications import VGG19,ResNet152V2
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D,AveragePooling2D,Input, Dropout, Flatten, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow import keras
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from skimage.io import imread
from skimage.transform import resize
from numpy import load,array,argmax
import numpy as np
import urllib.request
import os
import sys 
import requests
import pickle
import json
from collections import namedtuple

class ConfigLoader:

  def __init__(self,config_dict):
    Drive = namedtuple("Drive",config_dict["drive"].keys())
    self.drive = Drive(**config_dict["drive"])
    Local = namedtuple("Local",config_dict["local"].keys())
    self.local = Local(**config_dict["local"])

class My_Custom_Generator(Sequence) :
  
  def __init__(self, image_filenames, labels, batch_size,augmented=False) :
    self.image_filenames = image_filenames
    self.labels = labels
    self.batch_size = batch_size
    self.augmented = augmented
    
  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

  def augmentor(self,images):
    seq = iaa.Sequential([
    iaa.Crop(px=(1, 16), keep_size=False),
    iaa.GaussianBlur(sigma=(0, 3.0)),
    iaa.Multiply((1.2, 1.5))
    ])
    return seq.augment_images(images)
  
  def __getitem__(self, idx) :
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

    images = [imread(file_name) for file_name in batch_x]
    label_list = np.array(batch_y)

    if self.augmented:
      aug_img = self.augmentor(images)
      label_list = np.append(label_list, label_list)
      images= aug_img + images
    
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


def create_model():
    img_input = Input(shape=(256, 256, 3))

    # Block 1
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='block1_conv1', trainable=False)(img_input)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='block1_conv2', trainable=False)(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid', name='block1_pool', trainable=False)(x)

    #Block 2
    x = Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', name='block2_conv1',trainable=False)(x)
    x = Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', name='block2_conv2', trainable=False)(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid', name='block2_pool', trainable=False)(x)

    #Block 3
    x = Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', name='block3_conv1',trainable=False)(x)
    x = Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', name='block3_conv2', trainable=False)(x)
    x = Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', name='block3_conv3', trainable=False)(x)
    x = Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', name='block3_conv4', trainable=False)(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid', name='block3_pool', trainable=False)(x)

    #Block 4
    x = Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block4_conv1')(x)
    x = BatchNormalization(name='block4_bn1')(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block4_conv2')(x)
    x = BatchNormalization(name='block4_bn2')(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block4_conv3')(x)
    x = BatchNormalization(name='block4_bn3')(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block4_conv4')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid', name='block4_pool')(x)

    #Block 5
    x = Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block5_conv1')(x)
    x = BatchNormalization(name='block5_bn1')(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block5_conv2')(x)
    x = BatchNormalization(name='block5_bn2')(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block5_conv3')(x)
    x = BatchNormalization(name='block5_bn3')(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block5_conv4')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid', name='block5_pool')(x)
    
    #Other layers
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.7, name='dropout_1')(x)
    x = Dense(1000, activation='relu', name='fc2')(x)
    x = Dropout(0.7, name='dropout_2')(x)
    x = Dense(512, activation='relu', name='fc3')(x)
    x = Dropout(0.5, name='dropout_3')(x)
    x = Dense(3, activation='softmax', name='predictions')(x)
    
    model = Model(inputs=img_input, outputs=x)
    return model


def customJSONDecoder(customDict):
    return namedtuple('X', customDict.keys())(*customDict.values())

def get_from_pickle(file_id,destination):
  ## If no file_id is given, it will not download from drive
  if file_id or os.stat(destination).st_size == 0: 
    print("Downloading file from drive...")
    download_file_from_google_drive(file_id, destination)
    print("Saved file locally at {}".format(destination))
  pickled_object = pickle.load(open(destination,'rb'))
  return pickled_object


def compile_get_model(resnet=False,show_summary=False):
  model = create_model()
  opt = Adam(learning_rate=1e-4)
  model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
  return model

arguments = sys.argv 
EVALUATE_CONFIG = 'evaluate.json'
config = ConfigLoader(json.load(open(EVALUATE_CONFIG)))
if len(arguments)!=2:
  print('Command must be like:"python evaluate.py path_to_image.png"')
else:
  img_path = arguments[-1]

  ## Check that drive or local are not empty.
  assert(config.drive or config.local==True)

  if config.drive:
    print("Loading data from drive")
    drive =  config.drive 
    weights_id = drive.weights_id
    weights_destination = drive.weights_destination
    weights = get_from_pickle(weights_id, weights_destination)
    if drive.weights_only:

      ## get model from function
      model = compile_get_model()
    else:
      ## get model from pickle
      model_id = drive.model_id
      model_destination = drive.model_destination
      model = get_from_pickle(model_id)
      
    model.set_weights(weights)
  else:
    local = config.local
    if local.weights_only:
      model = compile_get_model()
      model.load_weights(local.weights_location)
    else:
      load_model(local.model_location)
  

  pred = model.predict(single_picture_loader(img_path))
  print(argmax(pred,axis=1)[0])
