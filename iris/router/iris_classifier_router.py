from fastapi import APIRouter,Body
from starlette.responses import JSONResponse
from skimage.io import imread
from skimage.transform import resize
from numpy import array,argmax,uint8
from tensorflow.keras.models import model_from_json
import base64

router = APIRouter()

def decode(base64_string):
    if isinstance(base64_string, bytes):
        base64_string = base64_string.decode("utf-8")
    imgdata = base64.b64decode(base64_string)
    img = imread(imgdata, plugin='imageio')
    return img

@router.post('/classify_iris')
def extract_name(file_data: str = Body(...)):

    with open("../../data/model_num.json", "r") as json_file:
        model = model_from_json(json_file.read())
    
    model.load_weights('../../data/modelo1_weights.h5')
    image = decode(file_data)
    image = resize(image, (224, 224,3))
    image = [image]
    image = array(image,dtype=uint8)/255.0
    image = [image for _ in range(3)]
    pred = model.predict(image, batch_size=1)
    pred = argmax(pred, axis = 1)
    return JSONResponse({'pred':pred})
