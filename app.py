import os
import numpy as np
from flask import Flask, request, jsonify, render_template

import keras
from keras.preprocessing import image
from keras import backend as K
from tensorflow.keras.models import load_model


#### flask setup ####
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

#### loading a keras model with flask ####
def loaded_model():
    global model
    model = load_model("Data/final_model.h5")
 

#### preprocess data function ####
def prepare_image(img):
    #convert image tp numpy array
    img = image.img_to_array(img)
    #scale the image 
    img_data = np.expand_dims(img, 0)
    datagen = image.ImageDataGenerator(rescale=1./255)
    final_data = datagen.flow(img_data)[0]
    return final_data

#### flask routes ####
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data")
def data():
    return render_template("data.html")

@app.route("/video")
def video():
    return render_template("video.html")

@app.route("/model")
def model():
    return render_template("model.html")

@app.route("/photo", methods=["GET","POST"])
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

            predicted_distraction = model.predict_classes(image_array)
            data["Prediction"]=str(predicted_distraction)
            data["Success"]=True

            return jsonify(data)
    return render_template("photo.html")

@app.route("/output")
def output():
    return render_template("output.html")

if __name__ == '__main__':
    loaded_model()
    app.run(debug=True)
