import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Input
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras.utils import to_categorical
import sys

### Import data
try:
    l = int(sys.argv[1])
    n = int(sys.argv[2])
except IndexError:
    l = 2
    n = 128


saveName = 'Output/DeepComparison-l={:d}-n={:d}.csv'.format(l,n)

###
data = pd.read_csv('Data/2018-06-01-NSW-SA1Input-Normalised.csv')

prediction = data.iloc[:,-1].values
training_data = data.iloc[:,1:-1].values  # Exclude SA2_MAINCODE_2016 and Category

no_folds = 5
inst = KFold(n_splits = no_folds, shuffle=True, random_state=125)


loss = []
val_loss = []
acc = []
val_acc = []
neurons = []
layers = []
rep = []

for i in range(no_folds):
    
    train_idx, test_idx = list(inst.split(np.arange(len(prediction))))[i]
        
    train_x = training_data[train_idx]
    train_y = prediction[train_idx]
    val_x = training_data[test_idx]
    val_y = prediction[test_idx]
    
    ### Build model
    K.clear_session()
    
    def buildModel(neurons, layers = 1):
        inputs = Input(shape = (train_x.shape[-1],))
        
        x = Dense(neurons, use_bias = False)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        
        for j in range(layers - 1):
            x = Dense(neurons, use_bias = False)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(0.5)(x)
            
        predictions = Dense(10, activation = 'softmax')(x)
        
        model = Model(inputs = inputs, outputs = predictions)
    
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        return model
    
    model = buildModel(neurons = n, layers = l)
    
    history = model.fit(train_x,
                        to_categorical(train_y, 10),
                        epochs=1000,
                        batch_size=64,
                        validation_data = (val_x, to_categorical(val_y, 10)))

    loss.append(history.history['loss'][-1])
    val_loss.append(history.history['val_loss'][-1])
    acc.append(history.history['acc'][-1])
    val_acc.append(history.history['val_acc'][-1])
    rep.append(i)
    layers.append(l)
    neurons.append(n)
    
    
    
loss = pd.DataFrame(loss, columns = ['Loss'])
val_loss = pd.DataFrame(val_loss, columns = ['Validation Loss'])
acc = pd.DataFrame(acc, columns = ['Accuracy'])
val_acc = pd.DataFrame(val_acc, columns = ['Validation Accuracy'])
rep = pd.DataFrame(rep, columns = ['Repeat'])
l = pd.DataFrame(layers, columns = ['Layers'])
n = pd.DataFrame(neurons, columns = ['Neurons'])

df = pd.concat([n, l, rep, loss, val_loss, acc, val_acc], axis = 1)
df.to_csv(saveName)
