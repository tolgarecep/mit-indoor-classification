import matplotlib.pyplot as plt
import os
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = models.load_model('cnn.h5')

test_dir = test_dir = './indoorCVPR_09/test'
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(256, 256), batch_size=67, 
                                                  class_mode='categorical')

img_test, label_test = test_generator.next()
labels = os.listdir(r'.\indoorCVPR_09\Images')

def show_prediction(i): 
    # show prediction for first batch's i-th sample
    pred = model.predict(img_test)[i]
    label_code = list(pred).index(pred.max())
    plt.imshow(img_test[i])
    print('Class:', labels[list(label_test[i]).index(1)], 'Predicted class:', labels[label_code])
    
show_prediction(0)
show_prediction(1)
show_prediction(2)
show_prediction(3)

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)