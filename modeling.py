import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

## Preprocessing with data Augmentation
train_dir = r'.\indoorCVPR_09\train'
val_dir = r'.\indoorCVPR_09\val'
test_dir = r'.\indoorCVPR_09\test'

datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2,
                                    height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                    horizontal_flip=True, fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255) # DO NOT AUGMENT VALIDATION OR TEST DATA## Modeling

train_generator = datagen.flow_from_directory(train_dir, target_size=(256, 256), batch_size=20,
                                                    class_mode='categorical')

val_generator = test_datagen.flow_from_directory(val_dir, target_size=(256, 256), batch_size=20,
                                                class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_dir, target_size=(256, 256), batch_size=20, 
                                                  class_mode='categorical')
## Modeling
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(67, activation='softmax'))

optimizer = optimizers.Adam(learning_rate=6e-5)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

checkpoint_cb = callbacks.ModelCheckpoint("cnn.h5", save_best_only=True)

history = model.fit(train_generator, steps_per_epoch=234, epochs=100, 
                    callbacks=[checkpoint_cb], validation_data=val_generator, validation_steps=30)

## Learning curve
with open('cnn-history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
        
history_dict = pickle.load(open('cnn-history', "rb"))

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()