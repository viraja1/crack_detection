from __future__ import division, print_function
# coding=utf-8

import os

import tensorflow as tf

MODEL_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

train_dir = os.path.join(MODEL_DATA_PATH, 'train')
validation_dir = os.path.join(MODEL_DATA_PATH, 'validation')
train_length = len(os.listdir(os.path.join(train_dir, 'crack'))) + len(os.listdir(os.path.join(train_dir, 'no_crack')))
validation_length = len(os.listdir(os.path.join(validation_dir, 'crack'))) + \
                    len(os.listdir(os.path.join(validation_dir, 'no_crack')))
batch_size = 300

model = tf.keras.models.Sequential([
 tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
 tf.keras.layers.MaxPool2D((2, 2)),
 tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
 tf.keras.layers.MaxPool2D((2, 2)),
 tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
 tf.keras.layers.MaxPool2D((2, 2)),
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(units=512, activation='relu'),
 tf.keras.layers.Dense(units=1, activation='sigmoid'),
])

model.summary()


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(128, 128), batch_size=batch_size, class_mode='binary')

validation_generator = train_datagen.flow_from_directory(validation_dir,
                                                         target_size=(128, 128), batch_size=batch_size,
                                                         class_mode='binary')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=int(train_length / batch_size),
      epochs=15,
      verbose=1,
      validation_data=validation_generator,
      validation_steps=int(validation_length / batch_size)
)

model.save(os.path.join(os.path.dirname(__file__), 'crack_detection.h5'))
