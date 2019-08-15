#!/usr/bin/env python3
""" A recreation of the tensorflow 2 tutorial MNIST exercise
located at https://www.tensorflow.org/tutorials/keras/basic_classification
"""

# Import Required Python Libraries
import tensorflow as tf
from matplotlib import pyplot as plt

# load the Fashion MNIST dataset from tf.keras
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Define the classification name associated with the 10 categories
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# # Displays a given image from dataset
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# Normalizes the data to be between 0 and 1 by dividing by highed value
train_images = train_images / 255.0
test_images = test_images / 255.0

# # Displays 10x10 grid of images along with thier labels
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# Builds the TF deep learning model
model = tf.keras.Sequential([
    # flattens pixel data from 2d to 1d
    tf.keras.layers.Flatten(input_shape=(28,28)),
    # creates first feature engineering/anlaysis layer
    tf.keras.layers.Dense(128, activation="relu"),
    # creates layer for final 10 classificaiton groups
    tf.keras.layers.Dense(10, activation="softmax")
    ])

# Compiles the models and sets trainings variables
model.compile(optimizer="adam",
loss = "sparse_categorical_crossentropy",
metrics=["accuracy"])

# train ("fit") the model with data & target/labels as input
model.fit(train_images, train_labels, epochs=5)

# Evaluates model retults accuracy using test dataset
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# Creates predictions for input test images
predictions = model.predict(test_images)


# displays image and text for given image color coded by prediction accuracy
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

# displays bar graph of the "confidence matrix" generated for input item
import numpy as np
def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# Displays image and confidence bar graft for item [0] in predictions list
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()
