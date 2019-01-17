import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import pickle
from imutils import paths
import os
from sklearn.preprocessing import LabelBinarizer

#from keras import backend as K

#K.set_image_dim_ordering('th')

img_width, img_height = 150, 150

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'Train_0.8'
validation_data_dir = 'Validation_0.8'
nb_train_samples = 29841
nb_validation_samples = 7481
epochs = 50
batch_size = 32

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    #np.save(open('bottleneck_features_train.npy', 'wb'),
    #        bottleneck_features_train)
    
    f = open('bottleneck_features_train.npy', 'wb')
    f.write(pickle.dumps(bottleneck_features_train))
    f.close()

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    #np.save(open('bottleneck_features_validation.npy', 'wb'),
    #        bottleneck_features_validation)
    f = open('bottleneck_features_validation.npy', 'wb')
    f.write(pickle.dumps(bottleneck_features_validation))
    f.close()

            
def train_top_model():
    #train_data = np.load(open('bottleneck_features_train.npy', encoding='utf8',errors='ignore'))
    train_data = pickle.loads(open('bottleneck_features_train.npy', "rb").read())
    print(len(train_data))
    lb = LabelBinarizer()
    
    train_imagePaths = sorted(list(paths.list_images(train_data_dir)))
    train_labels = []
    for imagePath in train_imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        train_labels.append(label)
    train_labels = np.array(train_labels[:len(train_data)])
    #train_labels = np.unique(train_labels)
    print(len(train_labels))
    train_labels = lb.fit_transform(train_labels)
    
    #train_labels = np.array(
    #    [0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))

    #validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_data = pickle.loads(open('bottleneck_features_validation.npy', "rb").read())
    print(len(validation_data))
    validation_imagePaths = sorted(list(paths.list_images(validation_data_dir)))
    #print(validation_imagePaths)
    validation_labels = []
    for imagePath in validation_imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        validation_labels.append(label)
    validation_labels = np.array(validation_labels[:len(validation_data)])
    #validation_labels = np.unique(validation_labels)
    print(len(validation_labels))
    validation_labels = lb.fit_transform(validation_labels)
    
    #validation_labels = np.array(
    #    [0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)


save_bottlebeck_features()
train_top_model()