import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model (make sure the path is correct)
model = tf.keras.models.load_model('mobilenetv2_model.keras')  # Adjust this path as necessary

# Load the image from the directory
img_path = "D:/chest_xray/chest_xray/test/NORMAL/IM-0010-0001.jpeg"  # Change this path as needed
test_image = tf.keras.utils.load_img(img_path, target_size=(256, 256))

# Display the loaded image
plt.imshow(test_image)
plt.axis('off')  # Hide axis
plt.show()  # Show the image

# Convert the loaded image into a NumPy array and expand its dimensions
test_image = tf.keras.utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Use the trained model to make a prediction on the input image
result = model.predict(test_image)

# Extract the probabilities of the input image belonging to each class
class_probabilities = result[0]

# Determine the class with the highest probability and print its label
if class_probabilities[0] > class_probabilities[1]:
    print("Normal")
else:
    print("Pneumonia")
