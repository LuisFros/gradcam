from tensorflow.keras.models import load_model

## Import custom functions and configuration
from loader import get_from_pickle, ConfigLoader

## Here import your model and functions for pre-processing
from model import compile_get_model


def main():
  EVALUATE_CONFIG = 'config.json'
  config = ConfigLoader(EVALUATE_CONFIG)
  
  config.download_all_files()

  model =  config.model
  weights =  config.weights

  if model.required:
    if model.pickle:
      model = get_from_pickle(model.location)
    else:
      model = load_model(model.location)
  else:
    model = compile_get_model()  

  ## required when model is not loaded with "load_model"
  if weights.required:
    if weights.pickle:
      weights = get_from_pickle(weights.location)
      model.set_weights(weights)
    else:
      model.load_weights(weights.location)
  
  ### Here goes the evaluation code for your own model (use model to predict)
  ## Ex: model.predict(...)
