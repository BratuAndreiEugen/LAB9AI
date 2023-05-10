from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential

def tool_sepiaCNN():
    print("\nSepia Filter Detection CNN :")

    datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = datagen.flow_from_directory(
        'C:\Proiecte SSD\Python\LAB9AI\sepiaData\\training_set',
        target_size=(100, 100),
        batch_size=32,
        class_mode='binary',
        shuffle=True
    )

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # default learning rate = 0.001
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_generator, epochs=5)

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        'C:\Proiecte SSD\Python\LAB9AI\sepiaData\\test_set',
        target_size=(100, 100),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(test_generator)

    print('Test accuracy:', test_acc)




def tool_sepiaANN():
    print("\nSepia Filter Detection ANN :")

    # Set the paths for the training and validation directories
    train_dir = "C:\Proiecte SSD\Python\LAB9AI\sepiaData\\training_set"
    test_dir = "C:\Proiecte SSD\Python\LAB9AI\sepiaData\\test_set"

    # Define image size and batch size
    img_size = (100, 100)
    batch_size = 16

    # Set up data generators for training and validation
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary')

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary')

    # Define the ANN model
    model = Sequential()
    model.add(Flatten(input_shape=img_size + (3,)))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    opt = Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    epochs = 10
    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator)

    # Evaluate the model on the validation set
    test_loss, test_acc = model.evaluate(test_generator)

    print('Test accuracy:', test_acc)