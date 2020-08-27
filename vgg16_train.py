from tensorflow.keras.preprocessing import image
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.models import save_model
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
BATCH = 40
LEARNING_RATE = 0.001
EPOCHS = 100

# create data using batch generators
def generate_data():
    # for training data
    train_gen = ImageDataGenerator()
    # for testing data
    validate_gen = ImageDataGenerator(rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    # train data and test data
    train_data = train_gen.flow_from_directory('PetImages/train/', target_size = (224, 224), batch_size = BATCH)
    validate_data = validate_gen.flow_from_directory('PetImages/validate/', target_size = (224, 224), batch_size = BATCH)
    return train_data, validate_data

# top 1 accuracy representation
def top_1_accuracy(in_gt, in_pred):
    return top_k_categorical_accuracy(in_gt, in_pred, k=1)

# create the layers of the model
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
    model.compile(optimizer = Adam(lr=LEARNING_RATE), 
                   loss ='categorical_crossentropy',
                   metrics = ['accuracy'])
    # get model summary
    model.summary()
    return model


# train the model
def train_model(model, train_data, validate_data):
    # select to train on cpu
    model.fit(train_data, validation_data =validate_data, validation_steps = validate_data.samples // BATCH  ,  steps_per_epoch = train_data.samples // BATCH, epochs = EPOCHS)
    # save model after training is complete
    with tf.device('/cpu:0'):
        cwd = os.getcwd()
        model.save("Results_second.h5")
    return model

# predict on test data
def predict_model(model, test_data):
    model.predict(test_data, steps=5)
    return
    
# driver function
def VGG16():
    # get data
    train_data, validate_data = generate_data()
    # generate and compile the model
    model = create_model()
    # train the model
    model = train_model(model, train_data, validate_data)
    # make predictions on test data 
    #predict_model(model, test_data)
    
    
VGG16()
