from keras.preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button, ttk, Frame
from PIL import Image, ImageTk

# Load the trained model
model = load_model('mobilenetv2_model.keras')

def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    image_array = image.img_to_array(img)
    image_array = np.expand_dims(image_array, axis=0)
    img_data = preprocess_input(image_array)

    # Get prediction
    prediction = model.predict(img_data)
    # For binary classification
    result = 'Person is affected with Pneumonia.' if prediction[0][0] > 0.5 else 'Person is safe.'
    return img, result

def open_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
    if file_path:
        img, result = predict_image(file_path)

        # Display the image
        img = img.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        img_label.config(image=img_tk)
        img_label.image = img_tk  # Keep a reference to avoid garbage collection

        # Show the prediction result
        result_label.config(text=f"Result: {result}")

def remove_image():
    # Clear the image and prediction result
    img_label.config(image="")
    img_label.image = None  # Remove reference to the image
    result_label.config(text="")

def close_app():
    root.destroy()

# Set up the GUI window
root = tk.Tk()
root.title("Pneumonia Detector")
root.geometry("500x700")
root.configure(bg="#2C3E50")

# Create a frame to organize content
frame = Frame(root, bg="#34495E", padx=10, pady=10)
frame.pack(expand=True, fill="both", padx=10, pady=10)

# Header Label
header = Label(frame, text="Pneumonia Detection", font=("Helvetica", 24, "bold"), fg="white", bg="#34495E")
header.pack(pady=15)

# Select Image Button
style = ttk.Style()
style.configure("TButton", font=("Helvetica", 14), padding=10)
select_button = ttk.Button(frame, text="Select Image", command=open_file)
select_button.pack(pady=10)

# Image Display Label
img_label = Label(frame, bg="#34495E")
img_label.pack(pady=20)

# Prediction Result Label
result_label = Label(frame, text="", font=("Helvetica", 18), fg="white", bg="#34495E")
result_label.pack(pady=20)

# Button Frame for Remove and Close Buttons
button_frame = Frame(frame, bg="#34495E")
button_frame.pack(pady=10)

# Remove Image Button
remove_button = ttk.Button(button_frame, text="Remove Image", command=remove_image)
remove_button.grid(row=0, column=0, padx=5)

# Close Button
close_button = ttk.Button(button_frame, text="Close", command=close_app)
close_button.grid(row=0, column=1, padx=5)

# Start the GUI event loop
root.mainloop()
