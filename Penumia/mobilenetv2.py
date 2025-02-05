from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob

# Paths for training and testing data
training_data = r'D:\chest_xray\chest_xray\train'
testing_data = r'D:\chest_xray\chest_xray\test'

# Image input shape
IMAGESHAPE = [224, 224, 3]

# Load MobileNetV2 model without top layers
mobilenet = MobileNetV2(input_shape=IMAGESHAPE, weights='imagenet', include_top=False)

# Freeze all the layers to use transfer learning
for layer in mobilenet.layers:
    layer.trainable = False

# Get the number of classes in the dataset
num_classes = len(glob(training_data + '/*'))

# Add custom layers on top of MobileNetV2
x = Flatten()(mobilenet.output)
prediction = Dense(num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=mobilenet.input, outputs=prediction)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Show the model summary
model.summary()

# Prepare ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load the training and testing data
training_set = train_datagen.flow_from_directory(
    training_data,
    target_size=(224, 224),
    batch_size=4,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    testing_data,
    target_size=(224, 224),
    batch_size=4,
    class_mode='categorical'
)

# Train the model
fitted_model = model.fit(
    training_set,
    validation_data=test_set,
    epochs=5,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

# Save the model in .keras format
model.save('mobilenetv2_model.keras')
print("Model saved as 'mobilenetv2_model.keras'")
