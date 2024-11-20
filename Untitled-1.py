from tensorflow.python.client import device_lib

# Load Data
import os
import cv2
import numpy as np

# Data Visualisation
import matplotlib.pyplot as plt

# Model Training
from tensorflow.keras import utils
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split


import kagglehub

path = kagglehub.dataset_download("grassknoted/asl-alphabet")

print("Path to dataset files:", path)

train_dir = path+'/asl_alphabet_train/asl_alphabet_train'

def get_data(data_dir) :
    images = []
    labels = []
    
    dir_list = os.listdir(data_dir)
    for i in range(len(dir_list)):
        print("Obtaining images of", dir_list[i], "...")
        for image in os.listdir(data_dir + "/" + dir_list[i]):
            img = cv2.imread(data_dir + '/' + dir_list[i] + '/' + image)
            img = cv2.resize(img, (32, 32))
            images.append(img)
            labels.append(i)
    
    return images, labels
        
X, y = get_data(train_dir)


classes = os.listdir(train_dir)
classes.sort()  # Ensure consistent order if necessary

def plot_sample_images():
    figure = plt.figure()
    plt.figure(figsize=(16, 5))

    for i in range(len(classes)):
        if i >= 29:  # Limit to 29 classes/images as in the original code
            break
        plt.subplot(3, 10, i + 1)
        plt.xticks([])
        plt.yticks([])
        path = f"{train_dir}/{classes[i]}/{classes[i]}1.jpg"
        if os.path.exists(path):  # Check if the image file exists
            img = plt.imread(path)
            plt.imshow(img)
            plt.xlabel(classes[i])
        else:
            print(f"Image not found: {path}")
        
    plt.show()

plot_sample_images()

def preprocess_data(X, y):
    np_X = np.array(X)
    normalised_X = np_X.astype('float32')/255.0
    
    label_encoded_y = utils.to_categorical(y)
    
    x_train, x_test, y_train, y_test = train_test_split(normalised_X, label_encoded_y, test_size = 0.1)
    
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = preprocess_data(X, y)

