# Step 1: Train your model and save it to the hard disk
# Assuming you have a trained model saved in the file 'my_model.h5'

# Step 2: Create environment by installing Flask
# You can install Flask and other required packages using: pip install flask gevent requests pillow

# Step 3: Build Keras REST API in a file named 'run_keras_server.py'
from flask import Flask, request, jsonify
import io
import numpy as np
from PIL import Image
from keras.applications import ResNet50, imagenet_utils
from keras.preprocessing.image import img_to_array

app = Flask(__name__)
model = None  # Placeholder for the Keras model

def load_model():
    global model
    model = ResNet50(weights="imagenet")

def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    return image

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if request.method == "POST":
        if request.files.get("image"):
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = prepare_image(image, target=(224, 224))

            preds = model.predict(image)
            results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []

            for (imagenetID, label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)

            data["success"] = True

    return jsonify(data)

# Step 4: Launch the service
if __name__ == "__main__":
    print("* Loading Keras model and Flask starting server... please wait until the server has fully started")
    load_model()
    app.run()

