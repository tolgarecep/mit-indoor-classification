## Remarks from previous step
# We'll use 5360 images for training, 670 for validation
# and 1340 for testing in this 67-class image classification task.
# These are very low sample sizes given the complexity of our problem,
# we'll try to overcome this after setting up a baseline.
# Here workflow will be as follows: building the baseline, data augmentation,
# feature extraction / fine-tuning.

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pickle

## Creating the batches
# target_size choice is arbitrary
# batch_size value must satisfy the following equation: #train-samples = batch_size * steps_per_epoch
# (steps_per_epoch will be defined later while fitting the model to training set)

train_dir = r'.\indoorCVPR_09\train'
val_dir = r'.\indoorCVPR_09\val'
test_dir = r'.\indoorCVPR_09\test'

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=20,
                                                    class_mode='categorical')
# Found 4680 images belonging to 67 classes.

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(150, 150), batch_size=20,
                                                class_mode='categorical')
# Found 670 images belonging to 67 classes.

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(150, 150), batch_size=20, 
                                                  class_mode='categorical')
# Found 1340 images belonging to 67 classes.

## Display samples
labels = list(train_generator.class_indices.keys())

# train_generator.next() is an iterator object
# which will iterator over batches of training data.
# img and label variables below represent first batch
# (it is randomly defined each time you run this file)
img, label = train_generator.next()

def show_sample(i):
    # show first batch's i-th sample
    plt.imshow(img[i])
    print(labels[list(label[i]).index(1)])
    
# show_sample(0)
# show_sample(1)
# show_sample(2)

## Building the baseline
model = models.Sequential()
model.add(layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(67, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(train_generator, steps_per_epoch=234, epochs=10, 
                    validation_data=val_generator, validation_steps=30)

# Save model
model.save('indoor-baseline.h5')

# Save history
with open('baseline-history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)