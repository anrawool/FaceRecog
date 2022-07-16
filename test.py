import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

cnn = tf.keras.models.load_model('saved_model/my_model')
test_image = image.load_img(
    'single_pred/IMG20220715173419.jpeg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image/255.0)


if result[0][0] > 0.5:
    prediction = 'Sarthak'
else:
    prediction = 'Elon'

print(prediction)
