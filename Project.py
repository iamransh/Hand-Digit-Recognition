import os
import cv2  # for computer vision and image processing
import matplotlib.pyplot as plt  # for visualization of the digits
import numpy as np  # for numpy arrays
import tensorflow as tf  # for deep learning and neural networks

mnist = tf.keras.datasets.mnist  # importing the mnist dataset from tensorflow keras
(x_train, y_train), (x_test, y_test) = (
    mnist.load_data()
)  # loading the mnist dataset into training and testing data

x_train = tf.keras.utils.normalize(x_train, axis=1)  # normalizing the training data
x_test = tf.keras.utils.normalize(x_test, axis=1)  # normalizing the testing data

model = tf.keras.models.Sequential()  # creating a sequential model
model.add(
    tf.keras.layers.Flatten(input_shape=(28, 28))
)  # flattening the input data(converts 2D pixel array to 1D pixel array)
model.add(
    tf.keras.layers.Dense(128, activation="relu")
)  # adding a dense layer with 128 neurons and relu activation function
model.add(
    tf.keras.layers.Dense(128, activation="relu")
)  # adding a dense layer with 128 neurons and relu activation function
model.add(
    tf.keras.layers.Dense(10, activation="softmax")
)  # adding a dense layer with 10 neurons and softmax activation function

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)  # compiling the model with adam optimizer, sparse categorical crossentropy loss function and accuracy as the metric

model.fit(x_train, y_train, epochs=3)  # training the model with 3 epochs

model.save("digits.model.keras")  # saving the model

model = tf.keras.models.load_model("digits.model.keras")  # loading the model

loss, accuracy = model.evaluate(x_test, y_test)  # evaluating the model

print(f"Loss: {loss}, Accuracy: {accuracy}")  # printing the loss and accuracy

image_number = 1

while os.path.isfile(f"digits/digit {image_number}.png"):
    try:
        image = cv2.imread(f"digits/digit {image_number}.png")[:, :, 0]
        image = np.invert(np.array([image]))
        prediction = model.predict(image)
        prediction = np.argmax(prediction)
        print(f"Prediction for digit {image_number}: {prediction}")
        plt.imshow(image[0], cmap=plt.cm.binary)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        image_number += 1
