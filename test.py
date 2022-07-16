import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

cnn = tf.keras.models.load_model('saved_model/my_model')
test_image = image.load_img(
    'single_pred/images.jpeg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image/255.0)


results = []

for i in result[0]:
    x = round(i, 2)
    results.append(x)

print(results)
if results[0] > 0.99:
    print("This is Bill")
elif results[1] > 0.99:
    print("This is Elon")
elif results[2] > 0.99:
    print("This is Sarthak")
else:
    print("This is somebody new...")
