from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from keras.metrics import top_k_categorical_accuracy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

BATCH = 32
LEARNING_RATE = 0.001
EPOCHS = 1000



# create data using batch generators
def generate_data():
    # for training datac
    train_gen = ImageDataGenerator(validation_split = 0.2)
    # train data and test data
    train_data = train_gen.flow_from_directory('Dataset/train/', target_size = (224, 224), subset = "training", batch_size = BATCH)
    validate_data = train_gen.flow_from_directory('Dataset/train/', target_size = (224, 224), subset = "validation", batch_size = BATCH)
    return train_data, validate_data


# create the layers of the model
def create_model():
    # create input 
    # epochs = 3
    inputs = layers.Input(shape=(224, 224, 3))
    # create layers
    x = layers.Conv2D(64, (3, 3), padding = "same", activation='relu')(inputs)
    x = layers.Conv2D(64, (3, 3), padding = "same", activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding = "same", activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), padding = "same", activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(x)
    x = layers.Conv2D(256, (3, 3), padding = "same", activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), padding = "same", activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), padding = "same", activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(x)
    x = layers.Conv2D(512, (3, 3), padding = "same", activation='relu')(x)
    x = layers.Conv2D(512, (3, 3), padding = "same", activation='relu')(x)
    x = layers.Conv2D(512, (3, 3), padding = "same", activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(x)
    x = layers.Conv2D(512, (3, 3), padding = "same", activation='relu')(x)
    x = layers.Conv2D(512, (3, 3), padding = "same", activation='relu')(x)
    x = layers.Conv2D(512, (3, 3), padding = "same", activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(x)
    # drop out 0.2
    # flatten
    x = layers.Flatten()(x)
    # create dense fully connected layers
    x = layers.Dense(4096, activation = 'relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(4096, activation = 'relu')(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(2, activation = 'softmax')(x)
    # define model
    model = Model(inputs=inputs, outputs=output)
    # compile model
    model.compile(optimizer = Adam(lr=LEARNING_RATE), 
                   loss = 'categorical_crossentropy',
                   metrics = ['accuracy'])
    # get model summary
    model.summary()
    return model


# train the model
def train_model(model, train_data, validate_data):
    earlystop = EarlyStopping(patience = 10)
    learning_rate_reduction = ReduceLROnPlateau(monitor ='val_acc', patience = 2, verbose = 1, factor = 0.5, min_lr = 0.001)
    checkpoint = ModelCheckpoint("Results.h5", monitor = 'val_acc', verbose=1, save_best_only = True, save_weights_only = False, mode = 'auto', period = 1)
    callback = [checkpoint, earlystop, learning_rate_reduction]
    # select to train on cpu
    model.fit_generator(train_data, steps_per_epoch = train_data.samples // BATCH, validation_data = validate_data, 
                        validation_steps = validate_data.samples // BATCH, epochs = EPOCHS, callbacks = callback)
    # save model after training is complete
    #model.save("Results.h5")
    return model

# driver function
def VGG16():
    # get data
    train_data, validate_data = generate_data()
    # generate and compile the model
    model = create_model()
    # train the model
    model = train_model(model, train_data, validate_data)
    
    
VGG16()
