from keras.models import Model
from keras.layers import Flatten, Dense
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
from glob import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Correcting paths
training_data = r'D:\chest_xray\chest_xray\train'
testing_data = r'D:\chest_xray\chest_xray\test'
val_data = r'D:\chest_xray\chest_xray\val'
# Image shape
IMAGESHAPE = [224, 224, 3]

# Load the pre-trained VGG16 model without the top fully connected layers
vgg_model = VGG16(input_shape=IMAGESHAPE, weights='imagenet', include_top=False)

# Freeze the layers of the pre-trained model
for each_layer in vgg_model.layers:
    each_layer.trainable = False

# Find the number of classes in the training set
classes = glob(training_data + '/*')

# Build the model
flatten_layer = Flatten()(vgg_model.output)
prediction = Dense(len(classes), activation='softmax')(flatten_layer)
final_model = Model(inputs=vgg_model.input, outputs=prediction)

# Display the model summary
final_model.summary()

# Compile the model
final_model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Prepare ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
testing_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess the training and testing datasets
training_set = train_datagen.flow_from_directory(training_data,  # Fixed path
                                                 target_size=(224, 224),
                                                 batch_size=4,
                                                 class_mode='categorical')

test_set = testing_datagen.flow_from_directory(testing_data,  # Fixed path
                                               target_size=(224, 224),
                                               batch_size=4,
                                               class_mode='categorical')

# Train the model
fitted_model = final_model.fit(
    training_set,
    validation_data=test_set,
    epochs=5,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

# Save the trained model
final_model.save('our_model.keras')
