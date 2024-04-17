import io
import os
import torch
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from torchvision import models, transforms
from PIL import Image
from flask import request

app = Flask(__name__)

# Load the VGG16 model
vgg16_model = load_model('vgg16_model.h5')

# Load the DenseNet121 model
densenet_model = load_model('vgg16_model.h5')

# Define the path to the ResNet model file
resnet_model_path = os.path.join(os.path.dirname(__file__), 'resnet18_model.pth')

# Load the ResNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5
resnet_model = models.resnet18(pretrained=False)
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, num_classes)
resnet_model.load_state_dict(torch.load(resnet_model_path, map_location=device))
resnet_model.eval()

# Define image transformations for ResNet model
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to predict using VGG16 model
def predict_vgg16(img):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    predicted_classes = vgg16_model.predict(img_array)
    predicted_class_index = np.argmax(predicted_classes)
    class_labels = ['No', 'Mild', 'Moderate', 'Severe', 'Proliferate']
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label
# Function to predict using VGG16 model
def predict_densenet(img):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    predicted_classes = vgg16_model.predict(img_array)
    predicted_class_index = np.argmax(predicted_classes)
    class_labels = ['No', 'Mild', 'Moderate', 'Severe', 'Proliferate']
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label
# Function to predict using ResNet model
def predict_resnet(image_path):
    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = resnet_model(img_tensor)
    predicted_class_index = torch.argmax(outputs).item()
    class_labels = ['Mild', 'No', 'Moderate', 'Severe', 'Proliferate']
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label

@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    print('hey')
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction="No file uploaded")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction="No file selected")

        if file:
            # Save the file to a temporary directory
            img_path = "uploads/image.png"
            file.save(img_path)

            selected_model = request.form.get('model')
            if selected_model == 'vgg16':
                prediction = predict_vgg16(image.load_img(img_path, target_size=(128, 128)))
                return render_template('result1.html', model='VGG16', prediction=prediction)
            elif selected_model == 'resnet':
                prediction = predict_resnet(img_path)
                return render_template('result2.html', model='ResNet', prediction=prediction)
            elif selected_model == 'densenet':
                prediction = predict_densenet(image.load_img(img_path, target_size=(128, 128)))
                return render_template('result3.html', model='DenseNet121', prediction=prediction)
            else:
                return render_template('result_error.html')  # Create a result_error.html template for handling errors

    return render_template('index.html')  # Render index.html if method is not POST or file not found
    
@app.route('/compare', methods=['POST'])
def compare():
    print('ethitto')
    if request.method == 'POST':
        if 'file' not in request.files:
            print('evidethi1')
            return render_template('index.html', prediction="No file uploaded")
            print('evidethi2')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction="No file selected")

        if file:
            # Save the file to a temporary directory
            img_path = "uploads/image.png"
            file.save(img_path)

            vgg16_prediction = predict_vgg16(image.load_img(img_path, target_size=(128, 128)))
            resnet_prediction = predict_resnet(img_path)
            densenet_prediction = predict_densenet(image.load_img(img_path, target_size=(128, 128)))
            
            print('evidethi2')
            return render_template('details.html', vgg16_result=vgg16_prediction, resnet_result=resnet_prediction ,densenet_result=vgg16_prediction)
            os.remove(img_path)

    #return render_template('index.html')  # Render index.html if method is not POST or file not found


@app.route('/compare', methods=['GET'])
def display_comparison():
    results = request.args.get('results')
    return render_template('compare.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
