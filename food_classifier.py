import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2

# Load data

(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
resized_image = cv2.resize(train_images[7], (280, 280), interpolation=cv2.INTER_CUBIC)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("This image is a:", class_names[train_labels[7]])

# Show the first image
plt.imshow(resized_image, cmap='gray')
plt.show()

