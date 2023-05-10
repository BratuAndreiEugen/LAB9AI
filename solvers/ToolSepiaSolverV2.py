# Import necessary libraries
import os
import cv2
import numpy as np
import keras
# Define the data directory, image dimensions, and train/test split ratio
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

def tool_sepia():
    IMG_WIDTH = 100
    IMG_HEIGHT = 100
    TEST_RATIO = 0.2

    # Load the dataset
    X = []
    y = []
    for root, dirs, files in os.walk('C:\Proiecte SSD\Python\LAB9AI\sepiaData\\training_set\\normal'):
        for file in files:
            if file.endswith('.jpg'):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                X.append(img)
                y.append(0)

    for root, dirs, files in os.walk('C:\Proiecte SSD\Python\LAB9AI\sepiaData\\training_set\sepia'):
        for file in files:
            if file.endswith('.jpg'):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                X.append(img)
                y.append(1)

    for root, dirs, files in os.walk('C:\Proiecte SSD\Python\LAB9AI\sepiaData\\test_set\\normal'):
        for file in files:
            if file.endswith('.jpg'):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                X.append(img)
                y.append(0)

    for root, dirs, files in os.walk('C:\Proiecte SSD\Python\LAB9AI\sepiaData\\test_set\sepia'):
        for file in files:
            if file.endswith('.jpg'):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                X.append(img)
                y.append(1)

    # Convert the data to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_RATIO, random_state=42)

    # Build the neural network model
    model = Sequential()
    # convolutional layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    # artificial
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print('Test accuracy:', accuracy)

    # Use the model to make predictions on new images
    # new_image = cv2.imread('/path/to/new/image.jpg')
    # new_image = cv2.resize(new_image, (IMG_WIDTH, IMG_HEIGHT))
    # new_image = np.expand_dims(new_image, axis=0)
    # prediction = model.predict(new_image)
    # if prediction > 0.5:
    #    print('The image has a sepia filter.')
    # else:
    #    print('The image does not have a sepia filter.')
