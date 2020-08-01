# Docker template for evaluating Tensorflow.Keras ML Models

The purpose of this tool is to provide an OpenSource option to evaluate Tensorflow models without Tensorflow installation issues.

- Local requirements:
    - Docker installed (see [Docker](https://docs.docker.com/desktop/))
    - docker-compose installed (see [docker-compose](https://docs.docker.com/compose/install/))
- Features: 
    - Custom packages can be installed in python image.
    - Download weights and/or model from Google Drive.
    - Load model from pickle, native keras formats or tensorflow.
    - Evaluate your model.

## Getting Started
1. First define the library requirements for loading and evaluating your model in `requierements.txt`. 
(Just as you would install with `pip3 install <library>`)
2. Run `docker-compose build` to create the image with all the requierements and environment for your evaluation.
3. Add your model code to `model.py` and import it in `evaluate.py`.
4. Upload weights or model to Googe Drive storage or locally if you prefer. (`.pkl`, `.h5` or `SavedModel` supported)
5. Update `config.json` according to your files.

### Example `config.json` 
-----
Create empty model and load weights from `.pkl`
```json
{
    "drive":[
            { "file_id":"FILE_ID_HERE", "destination":"model_weights.pkl"}
          ],
    "model":{
        "required":false,
        "location":"",
        "pickle":false
    },
    "weights":{
        "required":true,
        "location":"model_weights.pkl",
        "pickle":true
    }     
}
```
### 
## References:
 -  https://stackoverflow.com/a/39225039