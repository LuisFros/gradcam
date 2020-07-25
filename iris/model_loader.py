from tensorflow.keras.models import model_from_json

def get_model():
    with open("../data/model_num.json", "r") as json_file:
        model = model_from_json(json_file.read())
    model.load_weights('../data/modelo1_weights.h5')
    return model