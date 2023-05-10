import os
import numpy as np
from PIL import Image, ImageOps


def image_to_matrix(image_file, size = 20):
    # Open the image file
    img = Image.open(image_file)

    # Resize the image to 20x20
    img = img.resize((size, size))

    # Convert the image to grayscale
    img = img.convert('L')

    img = ImageOps.invert(img)

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Normalize the pixel values to be between 0 and 1
    img_array = img_array / 255.0

    # Return the 20x20 matrix
    return img_array


def getImageMatrices(path_to_images, size = 20):

    # Get a list of files in the directory
    image_files = os.listdir(path_to_images)

    matrices = []
    # Loop through each file and call image_to_matrix function
    for file in image_files:
        # Check if the file is an image file
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            # Get the full path to the image file
            image_file_path = os.path.join(path_to_images, file)

            # Call the image_to_matrix function to convert the image to a matrix
            image_matrix = image_to_matrix(image_file_path, size)
            matrices.append(image_matrix)

    return matrices


# m = getImageMatrices("C:\Proiecte SSD\Python\LAB9AI\sepiaData\\training_set\\normal")
# m = np.array(m)
# print(m[0])
# print(type(m))
# print(type(m[0]))
# # bune de trimis la ANN !!