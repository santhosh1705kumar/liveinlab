from fastapi import FastAPI, UploadFile, File, HTTPException
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

app = FastAPI()

# Load the TensorFlow models
plant_disease_model = tf.keras.models.load_model("models/plant_disease_model.h5")
soil_model = tf.keras.models.load_model("models/SoilNet_93_86.h5")

# Define class names for both models
plant_disease_class_names = ['early_rust', 'late_leaf_spot', 'nutrition_deficiency', 'healthy_leaf', 'early_leaf_spot', 'rust']
soil_class_names = ['Alluvial Soil', 'Black Soil', 'Clay Soil', 'Red Soil']

@app.post("/predict/groundnut")
async def predict_groundnut(file: UploadFile = File(...)):
    try:
        img = Image.open(BytesIO(await file.read()))
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict plant disease
        plant_disease_predictions = plant_disease_model.predict(img_array)
        predicted_disease_class = plant_disease_class_names[np.argmax(plant_disease_predictions)]

        return {"class": predicted_disease_class, "predictions": plant_disease_predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during plant disease prediction: {e}")

@app.post("/predict/soil")
async def predict_soil(file: UploadFile = File(...)):
    try:
        img = Image.open(BytesIO(await file.read()))
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict soil type
        soil_predictions = soil_model.predict(img_array)
        predicted_soil_class = soil_class_names[np.argmax(soil_predictions)]

        return {"class": predicted_soil_class, "predictions": soil_predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during soil prediction: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='localhost', port=8009)
