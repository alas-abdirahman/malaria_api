from flask import Flask, render_template, request, redirect, Response
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from tensorflow.keras.preprocessing import image
import datetime
import cv2

x = datetime.datetime.now()

# Initializing flask app
app = Flask(__name__)
model = load_model('./model/malaria_model.h5')
target_img = os.path.join(os.getcwd() , 'static/images')

print(model.summary())

# Setting the batch size, number of epochs, the image height and width parameters 
batch_size = 2000
epochs = 20 
IMG_HEIGHT = 128
IMG_WIDTH = 128

# Route for seeing a data
@app.route('/data')
def get_time():
	# Returning an api for showing in reactjs
	return {
		'Name':"geek",
		"Age":"22",
		"Date":x,
		"programming":"python"
		}

#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

# Function to load and prepare the image in right shape
def read_image(filename):
    img = load_img(filename, target_size=(128, 128))
    # x = image.img_to_array(img)
    x = np.expand_dims(img, axis=0)
    print("Afetr Image Read: {}".format(x.shape)); 
    x = preprocess_input(x)
    return x

def resizeImg(file):
    # Loading the image into memory 
    img = cv2.imread(file); 

    # Setting the dimensions for the loaded image to be converted into and displaying the shape of the image 
    print("Loaded Image Shape: {}".format(img.shape)); 
    dim = (128, 128); 

    # Resizing the image 
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA); 
    image = np.expand_dims(img, axis = 0); 
    print("After Image Expand: {}".format(img.shape)); 
    return image

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename): #Checking file format
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = resizeImg(file_path) #prepressing method
            print("After Image Preprocess: {}".format(img.shape)); 
            class_prediction=model.predict(img)
            print(str(class_prediction[0]) == "[0.]")
            classes_x=np.argmax(class_prediction,axis=1)
            if str(class_prediction[0]) == "[0.]":
              case = "Parasitized"
            elif str(class_prediction[0]) == "[1.]":
              case = "Uninfected"
            else: 
                case = "Unknown"

            print("caseeee: "+case)

            return case
        else:
            return "Unable to read the file. Please check file extension"
    else:
        return "Method not allowed"
	
# Running app
if __name__ == '__main__':
	app.run(debug=True)
