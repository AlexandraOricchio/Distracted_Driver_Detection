import os
import numpy as np
from flask import Flask, request, jsonify, render_template

import keras
from keras.preprocessing import image
from keras import backend as K


#### flask setup ####
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

#### loading a keras model with flask ####
def load_model():
    global model
    model = keras.models.load_model("model_here.h5")

#### preprocess data function ####
def prepare_image(img):
    img = image.resize(480,640)
    #convert image tp numpy array
    img = image.img_to_array(img)
    #scale the image pixels and invert the pixels
    img /= 255 
    img = 1 - img
    #flatten img to an array of pixels
    img_array = img.flatten().reshape(-1, 480*640)
    return img_array 

#### flask routes ####
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = {"Success": False}

    if request.method == "POST":
        if request.files.get("file"):
            # save file to uploads folder
            file = request.files["file"]
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # load image in with correct sizing 
            image_size = (480,640)
            im = image.load_img(filepath,target_size=image_size)
            # convert image to an array of values
            image_array = prepare_image(im)

            # get tensorflow default session & graph & make predictions
            with session.as_default():
                with session.graph.as_default():
                    predicted_distraction = model.predict_classes(image_array)[0]
                    data["Prediction"]=str(predicted_distraction)
                    data["Success"]=True

            return jsonify(data)
    return render_template("photo.html")

if __name__ == '__main__':
    app.run(debug=True)
