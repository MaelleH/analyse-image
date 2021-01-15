from keras.models import load_model
import keras.preprocessing.image as image

import os
import numpy as np

model = load_model('first_try.h5')

test_data_dir = 'data/test'

nb_test_samples = 0

img_width, img_height = 120, 120
i = 0

for root, dirs, files in os.walk(test_data_dir):
    nb_test_samples += len(files)

images = np.zeros((nb_test_samples, img_width, img_height, 3))

for root, dirs, files in os.walk(test_data_dir):
    for imageToLoad in files:
        img = image.load_img(os.path.join(root, imageToLoad), target_size=(img_width, img_height))
        images[i, :] = img
        i += 1

classes_predictions = np.argmax(model.predict(images), axis=-1)
