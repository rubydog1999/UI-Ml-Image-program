# Directory form:
# main_directory/
# ...class_a/
# ......a_image_1.jpg
# ......a_image_2.jpg
# ...class_b/
# ......b_image_1.jpg
# ......b_image_2.jpg
import os
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
from keras import backend as K
from keras.layers import Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator


# img_dir is absolute path
def ModelCreation(img_dir, img_width, img_height, epochs, batch_size, nb_train_samples, nb_validation_samples):
    #
    train_data_dir = img_dir + '/train'
    print("Train", train_data_dir)
    validation_data_dir = img_dir + '/validation'
    print("Val", validation_data_dir)
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    # Model definition
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    # This is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    return model
    # save model and architecture to single file


# model_Name="Model"+img_dir
# model_Name.replace("/","-")
# model.save(model_Name+".h5")
# For load the model:
# model = load_model('model.h5')


# img_dir="/C701-2" # it contains Good&Pass and Fail folders.
# source=os.getcwd()+img_dir
# filesList=os.listdir(img_dir+ "\\" + folder)

###
current_dir = os.getcwd()
img_dir = "\\IMG"
source = os.getcwd()
# os.chdir(os.getcwd()+img_dir)
dirList = os.listdir(source)
count = 0
epochs = 50
batch_size = 32
data = []

for folder in dirList:
    img_dir = source + '/C527-2'  # it is something like: .....\\C701-2
    if (os.path.isfile(source + "\\" + folder) == False) and (folder != "Backup") and (("._" in folder) == False):
        print("Folder:", folder)
        # To read one image of this folder for knowing image size

        fileList = os.listdir('C527-2/train/GOOD')
        img_sample = cv2.imread('C527-2/train/GOOD/' + fileList[0])
        img_width, img_height = img_sample.shape[:2]
        #
        nb_train_samples = len(os.listdir('C527-2/train/GOOD')) + len(os.listdir('C527-2/train/FAIL'))
        nb_validation_samples = len(os.listdir('C527-2/validation/GOOD')) + len(os.listdir('C527-2/validation/FAIL'))

        # Call NN model
        summary = [folder, img_width, img_height, nb_train_samples, nb_validation_samples]
        data.append(summary)
        count = count + 1
        m1 = ModelCreation(img_dir, img_width, img_height, epochs, batch_size, nb_train_samples, nb_validation_samples)

        # save model and architecture to single file
        # model_Name.replace("/","-")
        print("Saving:", "Model-" + folder + ".h5")
        m1.save("Modell-"+ folder + ".h5")
        # For load the model:
        # model = load_model('model.h5')

# Testing

img1 = cv2.imread('/C527-2/validation/GOOD/AOI307;20191105143455;A2C1304110350;A;5501247094470133;C527-2;1;Multi-Lighting;74448,70734;1658,1658;GOOD.jpg')
img2 = cv2.imread('C527-2/validation/FAIL/AOI307;20200131183129;A2C1304110450;A;8501247000521685;C527-2;1;Multi-Lighting;74448,70734;1658,1658;FAIL-2.jpg')
img1 = cv2.resize(img1, (98, 98))
img1 = np.reshape(img1, [1, 98, 98, 3])
img2 = cv2.resize(img2, (98, 98))
img2 = np.reshape(img2, [1, 98, 98, 3])

pred1 = m1.predict_classes(img1)
pred2 = m1.predict_classes(img2)
pred2 = m1.predict_classes(img2)
