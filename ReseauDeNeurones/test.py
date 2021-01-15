from keras.models import load_model
import keras.preprocessing.image as image
from shutil import copy
import os
import numpy as np

model = load_model('first_try.h5')

test_data_dir = 'data/test'
result_data_dir = 'data/results'

nb_test_samples = 0

img_width, img_height = 120, 120
i = 0

for root, dirs, files in os.walk(test_data_dir):
    nb_test_samples += len(files)

images = np.zeros((nb_test_samples, img_width, img_height, 3))
images_path = []
for root, dirs, files in os.walk(test_data_dir):
    for imageToLoad in files:
        img = image.load_img(os.path.join(root, imageToLoad), target_size=(img_width, img_height))
        images[i, :] = img
        images_path.append(imageToLoad)
        i += 1

classes_predictions = np.argmax(model.predict(images), axis=-1)

for root, dirs, files in os.walk(result_data_dir):
    for file in files:
        os.remove(os.path.join(root, file))

for i in range(len(classes_predictions)):
    classe_result_dir = os.path.join(result_data_dir, str(classes_predictions[i]))
    if not os.path.isdir(classe_result_dir):
        os.mkdir(classe_result_dir)
    copy(os.path.join(test_data_dir, images_path[i]), os.path.join(classe_result_dir, images_path[i]))


print(classes_predictions)





