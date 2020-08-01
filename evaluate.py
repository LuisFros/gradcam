from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file
from tensorflow.keras.utils import Sequence
# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from numpy import load,array,argmax
import urllib.request
import sys 


from model import compile_get_model,single_picture_loader,generate_grad_cam
from loader import get_from_pickle, ConfigLoader

EVALUATE_CONFIG = 'config.json'

arguments = sys.argv 
config = ConfigLoader(EVALUATE_CONFIG)

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
      model = get_from_pickle(model_id, model_destination)
      
    model.set_weights(weights)
  else:
    local = config.local
    if local.weights_only:
      model = compile_get_model()
      model.load_weights(local.weights_location)
    else:
      load_model(local.model_location)
  
  ### Here goes the evaluation code for your own model
  
  pred = model.predict(single_picture_loader(img_path))
  print(argmax(pred,axis=1)[0])

  generate_grad_cam(img_path,model,"gradcam.jpg")
