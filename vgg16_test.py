from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from keras.metrics import top_k_categorical_accuracy
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

BATCH = 64

def generate_data():
    # for testing data
    test_gen = ImageDataGenerator(rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    test_data = test_gen.flow_from_directory('PetImages/test/', target_size = (224, 224), batch_size = BATCH)
    return test_data




# predict on test data
def predict_model(test_data):
    model = keras.models.load_model("Results_second.h5")
    print(model.predict(test_data, steps=5))
    test_scores = model.evaluate(test_data, verbose=2)
    return
    
    
    
test_data = generate_data()
predict_model(test_data)
