import pickle
import json
from collections import namedtuple
import requests
import os

class ConfigLoader:

    def __init__(self,evaluate_path):
        config_dict = json.load(open(evaluate_path))
        modelTuple = namedtuple("modelTuple",config_dict["model"].keys())
        self.model = modelTuple(**config_dict["model"])
        Weights = namedtuple("Weights",config_dict["weights"].keys())
        self.weights = Weights(**config_dict["weights"])
        self.files = config_dict["drive"]
        
        self.check_config()
    def download_all_files(self):
        for file_obj in self.files:
            file_id = file_obj["file_id"]
            destination = file_obj["destination"]
            if os.stat(destination).st_size == 0: 
                print("Downloading file from drive...")
                download_file_from_google_drive(file_id, destination)
                print("Saved file locally at {}".format(destination))

    def check_config(self):
        assert(not(self.model.required) and self.weights.required == True,"If model defined from scratch, weights need to be given")




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

def customJSONDecoder(customDict):
    return namedtuple('X', customDict.keys())(*customDict.values())

def get_from_pickle(destination):
  pickled_object = pickle.load(open(destination,'rb'))
  return pickled_object