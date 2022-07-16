import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)  # Parameters for Datapreprocessing

training_set = train_datagen.flow_from_directory(
    'Dataset/Training_Set/',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')  # Training Set

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    'Dataset/Test_Set/',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')  # Test Set

# Building The CNN

cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3,
                               activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=3, activation='softmax'))


# # Compiling the CNN

cnn.compile(optimizer='adam', loss='categorical_crossentropy',
            metrics=['accuracy'])

cnn.fit(x=training_set, validation_data=test_set, epochs=5)

test_image = image.load_img(
    'single_pred/ab.jpeg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image/255.0)


if result[0][0] > 0.5:
    prediction = 'Sarthak'
else:
    prediction = 'Elon'


print(prediction)
cnn.save('saved_model/my_model')
