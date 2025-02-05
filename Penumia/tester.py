from keras.preprocessing import image
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input  # Use the appropriate preprocess function for MobileNetV2
import numpy as np
import os

# Load the trained model
model = load_model('mobilenetv2_model.keras')  # Ensure this path is correct

# Specify the image path
img_path = 'D:/chest_xray/test/PNEUMONIA/person100_bacteria_480.jpeg'

# Check if the image file exists
if os.path.isfile(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    image_array = image.img_to_array(img)  # Convert the image to an array
    image_array = np.expand_dims(image_array, axis=0)  # Expand dimensions to fit model input
    img_data = preprocess_input(image_array)  # Preprocess the image data

    # Make prediction
    prediction = model.predict(img_data)

    # Print prediction results
    if prediction[0][0] > prediction[0][1]:  # Assuming binary classification
        print('Person is safe.')
    else:
        print('Person is affected with Pneumonia.')

    print(f'Predictions: {prediction}')
else:
    print(f"File not found: {img_path}")
