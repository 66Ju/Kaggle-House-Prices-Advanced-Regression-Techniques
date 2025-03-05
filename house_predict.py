import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.models import Sequential
from keras import layers
from keras import optimizers
from keras.layers import BatchNormalization, Dropout, Dense
from keras.callbacks import ModelCheckpoint
import os
import math

df_train = pd.read_csv("C:/Users/penny/Downloads/train.csv")
df_test = pd.read_csv("C:/Users/penny/Downloads/test.csv")

df_train.drop("Id", inplace = True, axis = 1)
df_test.drop("Id", inplace = True, axis = 1)

object_columns = df_train.select_dtypes(include = ['object']).columns

for col in object_columns:
    df_train[col] = LabelEncoder().fit_transform(df_train[col].astype(str))
    df_test[col] = LabelEncoder().fit_transform(df_test[col].astype(str))

train_corr = df_train.select_dtypes(include = ['number']).corr()

plt.figure(figsize = (20, 20))  
sns.heatmap(train_corr, annot = True, fmt = ".2f", cmap = "Blues", vmax = 1, vmin = -1,
            linewidths = 0.5, annot_kws = {"size": 8}, cbar_kws = {"shrink": 0.8})

plt.xticks(rotation = 90, fontsize = 10)
plt.yticks(rotation = 0, fontsize = 10)  
plt.title("Correlation Heatmap", fontsize = 15)
plt.show()

high_corr = train_corr.index[abs(train_corr["SalePrice"]) > 0.6]

for i in df_train.columns:
    if i not in high_corr:
        df_train = df_train.drop(i, axis = 1)

train_targets = df_train["SalePrice"].values
train_data = df_train.drop(columns = ["SalePrice"])

X_train, X_validation, Y_train, Y_validation = train_test_split(train_data, train_targets, test_size = 0.2, random_state = 0)

X_train_dataset = X_train.values
X_validation_dataset = X_validation.values

normalize = preprocessing.StandardScaler()
X_train_normal_data = normalize.fit_transform(X_train_dataset)
X_validation_normal_data = normalize.fit_transform(X_validation_dataset)

def model():
    model = Sequential()
    
    model.add(layers.Dense(1024, kernel_initializer = 'random_normal', activation = 'relu', input_shape = (X_train_normal_data.shape[1],)))  
    model.add(Dropout(0.3))
    
    model.add(layers.Dense(1024, kernel_initializer = 'random_normal', activation = 'relu'))
    model.add(Dropout(0.3))
    
    model.add(layers.Dense(512, kernel_initializer = 'random_normal', activation = 'relu'))
    model.add(Dropout(0.3))
    
    model.add(layers.Dense(256, kernel_initializer = 'random_normal', activation = 'relu'))
    model.add(Dropout(0.3))
    
    model.add(layers.Dense(128, kernel_initializer = 'random_normal', activation = 'relu'))
    model.add(Dropout(0.3))
    
    model.add(layers.Dense(32, kernel_initializer = 'random_normal', activation = 'relu'))
    model.add(Dropout(0.3))
    
    model.add(layers.Dense(16, kernel_initializer = 'random_normal', activation = 'relu'))
    model.add(Dropout(0.3))
    
    model.add(layers.Dense(1, kernel_initializer = 'random_normal', activation = 'linear'))
    
    adam = optimizers.Adam(learning_rate = 0.001)
    model.compile(optimizer = adam, loss = 'mae')
    
    return model

call = ModelCheckpoint('good.weights.h5',
                       monitor = 'val_loss',
                       verbose = 0,
                       save_best_only = True,
                       save_weights_only = True,
                       mode = 'auto',
                       save_freq = 1)

model = model()
history = model.fit(X_train_normal_data, Y_train,
                    validation_data = (X_validation_normal_data, Y_validation),
                    callbacks = [call],
                    epochs = 600,
                    batch_size = 512,
                    verbose = 1)
model.save_weights("good.weights.h5")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc = 'upper right')
plt.show()

for i in df_test.columns:
    if i not in high_corr:
        df_test = df_test.drop(i, axis = 1)


X_test_dataset = df_test.values

normalize = preprocessing.StandardScaler()
X_test_normal_data = normalize.fit_transform(X_test_dataset)

model.load_weights('good.weights.h5')
pred = model.predict(X_test_normal_data)

with open("C:/Users/penny/Downloads/house_predict.csv", 'w') as f:
    f.write('Id,SalePrice\n')
    for i in range(len(pred)):
        f.write(str(i + 1461) + ',' + str(float(pred[i])) + '\n')
        
print(pred)
