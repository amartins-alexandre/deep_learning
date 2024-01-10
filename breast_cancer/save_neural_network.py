import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout

inputs = pd.read_csv("data/breast_inputs.csv")
outputs = pd.read_csv("data/breast_output.csv")

sequential = Sequential()
sequential.add(Dense(units=8, activation='relu', kernel_initializer='normal', input_dim=30))
sequential.add(Dropout(rate=0.2))
sequential.add(Dense(units=8, activation='relu', kernel_initializer='normal'))
sequential.add(Dropout(rate=0.2))
sequential.add(Dense(units=1, activation='sigmoid'))
sequential.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
sequential.fit(inputs, outputs, batch_size=10, epochs=100)

data_json = sequential.to_json()
with open("data/model_breast.json", "w") as json_file:
    json_file.write(data_json)

sequential.save_weights("data/weights_breast.h5")
