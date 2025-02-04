from fastapi import FastAPI, File, UploadFile
import uvicorn 
import numpy as np 
from io import BytesIO
from PIL import Image
import tensorflow as tf 


app = FastAPI()

model_name = "mash_net_2"
MODEL = tf.keras.models.load_model(f"../models/{model_name}")
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

@app.get("/health")
async def health_check():
    return {"status": "ok"}


def read_file_as_img(data) -> np.ndarray:
   return np.array(Image.open(BytesIO(data)))


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
):
    img = read_file_as_img(await file.read())
    img_batch = np.expand_dims(img, 0)
    
    prediction = MODEL.predict(img_batch)
    
    return {"filename": file.filename}


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host='localhost')