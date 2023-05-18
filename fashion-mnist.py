import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
import os

# Load the Fashion-MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Preprocess the data
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
x_train = x_train / 255.0
x_test = x_test / 255.0

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Check if saved model exists, and load it if available
saved_model_path = 'fashion_mnist_model.h5'
if os.path.exists(saved_model_path):
    model.load_weights(saved_model_path)
    print("Saved model loaded successfully!")
else:
    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1)
    # Save the trained model
    model.save_weights(saved_model_path)
    print("Model trained and saved!")

# Define the class labels
class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load and preprocess the input image
img_path = 'input_image.jpg'  # Replace with your own image path
img = image.load_img(img_path, target_size=(28, 28))
if img.mode != 'L':
    img = img.convert('L')
img = img.resize((28, 28))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Predict the garment type
predictions = model.predict(img_array)
predicted_label = np.argmax(predictions[0])
predicted_class = class_labels[predicted_label]

print("Predicted garment type:", predicted_class)
