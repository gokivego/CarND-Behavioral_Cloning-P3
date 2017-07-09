import pandas as pd
from keras import optimizers
from sklearn import model_selection
from generator import dataGenerator
from keras.models import Sequential
from keras.layers import core, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

def model():
    model1 = Sequential()
    model1.add(Convolution2D(16, 3, 3, input_shape=(32, 128, 3), activation='relu'))
    model1.add(MaxPooling2D(pool_size=(2, 2)))
    model1.add(Convolution2D(32, 3, 3, activation='relu'))
    model1.add(MaxPooling2D(pool_size=(2, 2)))
    model1.add(Convolution2D(64, 3, 3, activation='relu'))
    model1.add(MaxPooling2D(pool_size=(2, 2)))
    model1.add(core.Flatten())
    model1.add(core.Dropout(0.5))
    model1.add(core.Dense(200))
    model1.add(Activation('relu'))
    model1.add(core.Dropout(0.25))
    model1.add(core.Dense(50))
    model1.add(Activation('relu'))
    model1.add(core.Dropout(0.25))
    model1.add(core.Dense(10))
    model1.add(Activation('relu'))
    model1.add(core.Dense(1))
    return model1

if __name__ == '__main__':
    data_dir = './data/'
    data = pd.read_csv(data_dir + 'driving_log_balanced.csv')
    data_train, data_valid = model_selection.train_test_split(data, test_size=.2)
    model = model()
    dG = dataGenerator(data_dir + 'IMG/')
    model.compile(optimizer=optimizers.Adam(lr=1e-04), loss='mean_squared_error')
    history = model.fit_generator(
        dG.dataGen(data_train),
        samples_per_epoch=data_train.shape[0]*2,
        nb_epoch=25,
        validation_data=dG.validationGen(data_valid),
        nb_val_samples=data_valid.shape[0]
    )
    model.save('model1.h5')