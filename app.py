import os
import numpy as np
from flask import Flask, request, jsonify, render_template

import keras
from keras.preprocessing import image
from keras import backend as K
from tensorflow.keras.models import load_model

#### Converting Number Codes to Distraction Type ####
def distractiontype(i):
    switcher={
        0:'safe driving',
        1:'texting right',
        2:'talking on the phone right',
        3:'texting left',
        4:'talking on the phone left',
        5:'operating the radio',
        6:'drinking',
        7:'reaching behind',
        8:'hair and makeup',
        9:'talking to passenger'
    }
    return switcher.get(i,"Invalid day of week")

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
            #### make sure image is in correct format and give unique file name
            # if filename.endswith('.jpg'):
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

            # return jsonify(data)
    return render_template("photo.html",data=data)

@app.route("/output")
def output():
    return render_template("output.html")

if __name__ == '__main__':
    loaded_model()
    app.run(debug=True)
