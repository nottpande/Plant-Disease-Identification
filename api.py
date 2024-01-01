from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Loading the model
model = tf.keras.models.load_model('plant_identification.keras')
classes = ['Pepper__bell___Bacterial_spot','Pepper__bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy','Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot','Tomato_Spider_mites_Two_spotted_spider_mite','Tomato__Target_Spot','Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato__Tomato_mosaic_virus','Tomato_healthy']

@app.get("/ping")
async def ping ():
    return "The server has been loaded."

def convert(data):
    img_array = np.array(Image.open(BytesIO(data)))
    return img_array


@app.post("/predict")
async def prediction(file: UploadFile): #image of the plant
    #converting the image to bytes
    nparray = convert(await file.read())
    img_batch = np.expand_dims(nparray,axis=0)

    prediction = model.predict(img_batch)
    predicted_class = classes[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])

    return {
        'disease' : predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8800)
    