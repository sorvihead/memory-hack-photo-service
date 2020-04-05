import base64
import json
from dataclasses import dataclass
from pathlib import Path

from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Activation, Dense, Dropout, Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf

CUR_DIR = Path(__file__).resolve().parent


class ColorizationService:
    def __init__(self):
        self._path_to_model = CUR_DIR / 'model.json'
        self._path_to_weights = CUR_DIR / 'model.h5'
        self._path_to_train_data = CUR_DIR / 'data' / 'testdata' / 'Train'
        self._path_to_validate_data = CUR_DIR / 'data' / 'testdata' / 'Validate'
        self._path_to_output = CUR_DIR / 'result'
        self._path_to_input = CUR_DIR / 'input'
        self._batch_size = 10
        self._model = self._make_model()
        self._validate_model()

    def _get_images(self):
        X = []
        for filename in os.listdir(self._path_to_train_data):
            X.append(img_to_array(load_img(self._path_to_train_data / filename)))
        X = np.array(X, dtype=float)
        return X

    def _set_up_train_and_test_data(self, X):
        split = int(0.95 * len(X))
        Xtrain = X[:split]
        Xtrain = 1.0 / 255 * Xtrain
        return Xtrain

    batch_size = 10

    def _image_a_b_gen(self, datagen, Xtrain):
        for batch in datagen.flow(Xtrain, batch_size=self._batch_size):
            lab_batch = rgb2lab(batch)
            X_batch = lab_batch[:, :, :, 0]
            Y_batch = lab_batch[:, :, :, 1:] / 128
            yield (X_batch.reshape(X_batch.shape + (1,)), Y_batch)

    def _make_model(self):

        X = self._get_images()
        split = int(0.95 * len(X))
        XTrain = X[:split]
        XTrain = 1.0 / 255 * XTrain
        model = Sequential()
        model.add(InputLayer(input_shape=(256, 256, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.compile(optimizer='rmsprop', loss='mse')
        if self._path_to_weights:
            model.load_weights(str(self._path_to_weights))
            return model
        datagen = ImageDataGenerator(
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=20,
            horizontal_flip=True)

        model.fit_generator(self._image_a_b_gen(datagen, XTrain), epochs=100, steps_per_epoch=10)
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("model.h5")
        Xtest = rgb2lab(1.0 / 255 * X[split:])[:, :, :, 0]
        Xtest = Xtest.reshape(Xtest.shape + (1,))
        Ytest = rgb2lab(1.0 / 255 * X[split:])[:, :, :, 1:]
        Ytest = Ytest / 128
        print(model.evaluate(Xtest, Ytest, batch_size=self._batch_size))
        return model

    def _validate_model(self):
        color_me = []
        for filename in os.listdir(self._path_to_validate_data):
            color_me.append(img_to_array(load_img(self._path_to_validate_data / filename)))
        color_me = np.array(color_me, dtype=float)
        color_me = rgb2lab(1.0 / 255 * color_me)[:, :, :, 0]
        color_me = color_me.reshape(color_me.shape + (1,))

        # Test model
        output = self._model.predict(color_me)
        output = output * 128

        # Output colorizations
        for i in range(len(output)):
            cur = np.zeros((256, 256, 3))
            cur[:, :, 0] = color_me[i][:, :, 0]
            cur[:, :, 1:] = output[i]
            imsave(f"{self._path_to_output}/img_" + str(i) + ".png", lab2rgb(cur))

    def colorize(self, photo_request):
        photo_bytes = base64.b64decode(photo_request.base64_string)
        with open(self._path_to_input / f'{photo_request.chat_id}.jpg', 'wb') as f:
            f.write(photo_bytes)
        self._colorize_photo(photo_request.chat_id)
        return self._convert_to_base64_string(photo_request.chat_id)

    def _convert_to_base64_string(self, chat_id):
        with open(self._path_to_output / f'{chat_id}.png', 'rb') as f:
            base64_string = base64.b64encode(f.read()).decode()
        return base64_string

    def _scale_image(self, filename, width=256, height=256):
        original_image = Image.open(self._path_to_input / filename)
        transformed = original_image.resize((width, height), Image.ANTIALIAS)
        transformed.save(self._path_to_input / filename)

    def _colorize_photo(self, chat_id):
        self._scale_image(f'{chat_id}.jpg')
        image = [img_to_array(load_img(self._path_to_input / f'{chat_id}.jpg'))]
        image = np.array(image, dtype=float)
        color_me = rgb2lab(1.0 / 255 * image)[:, :, :, 0]
        color_me = color_me.reshape(color_me.shape + (1,))

        output = self._model.predict(color_me)
        output = output * 128

        for i in range(len(output)):
            cur = np.zeros((256, 256, 3))
            cur[:, :, 0] = color_me[i][:, :, 0]
            cur[:, :, 1:] = output[i]
            imsave(self._path_to_output / f'{chat_id}.png', lab2rgb(cur))
