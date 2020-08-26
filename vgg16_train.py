from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# create data using batch generators
def generate_data():
    # for training data
    train_gen = ImageDataGenerator()
    # for testing data
    test_gen = ImageDataGenerator(rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    # train data and test data
    train_data = train_gen.flow_from_directory('train/', target_size = (224, 224), batch_size = 64)
    test_data = test_gen.flow_from_directory('test/', target_size = (224, 224), batch_size = 64)
    return train_data, test_data


def create_model():
    # create input 
    inputs = layers.Input(shape=(224, 224, 3))
    # create layers
    x = layers.Conv2D(64, (3, 3), padding = "same", activation='relu')(inputs)
    x = layers.Conv2D(64, (3, 3), padding = "same", activation='relu')(x)
    x = layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding = "same", activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), padding = "same", activation='relu')(x)
    x = layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(x)
    x = layers.Conv2D(256, (3, 3), padding = "same", activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), padding = "same", activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), padding = "same", activation='relu')(x)
    x = layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(x)
    x = layers.Conv2D(512, (3, 3), padding = "same", activation='relu')(x)
    x = layers.Conv2D(512, (3, 3), padding = "same", activation='relu')(x)
    x = layers.Conv2D(512, (3, 3), padding = "same", activation='relu')(x)
    x = layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(x)
    x = layers.Conv2D(512, (3, 3), padding = "same", activation='relu')(x)
    x = layers.Conv2D(512, (3, 3), padding = "same", activation='relu')(x)
    x = layers.Conv2D(512, (3, 3), padding = "same", activation='relu')(x)
    x = layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(x)
    # flatten
    x = layers.Flatten()(x)
    # create dense fully connected layers
    x = layers.Dense(4096, activation = 'relu')(x)
    x = layers.Dense(4096, activation = 'relu')(x)
    output = layers.Dense(2, activation = 'softmax')(x)
    # define model
    model = Model(inputs=inputs, outputs=output)
    # compile model
    model.compile(optimizer="adam", loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    # get model summary
    model.summary()
    return model


# train the model
def train_model(train_data):
    # select to train on cpu
    with tf.device('/CPU:0'):
        model.fit(train_data,epochs = 3)
    # save model after training is complete
    model.save("model/")
    return model

# predict on test data
def predict_model()model:
    model.predict(test_data, steps=5)
    return
    
# driver function
def VGG16():
    # get data
    train_data, test_data = generate_data()
    # generate and compile the model
    model = create_model()
    # train the model
    model = train_model(train_data)
    # make predictions on test data 
    predict_model(model)
    
    
VGG16()