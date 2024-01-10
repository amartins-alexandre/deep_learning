import numpy as np
import pandas as pd
from keras.models import model_from_json

file = open('data/model_breast.json', 'r')
network_structure = file.read()
file.close()

sequential = model_from_json(network_structure)
sequential.load_weights('data/weights_breast.h5')

input_data = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178, 0.20, 0.05, 1098, 0.87, 4500, 145.2,
                        0.005, 0.04, 0.05, 0.015, 0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185, 0.84, 158, 0.363
                        ]])

predictions = sequential.predict(input_data)

inputs = pd.read_csv("data/breast_inputs.csv")
outputs = pd.read_csv("data/breast_output.csv")
sequential.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
result = sequential.evaluate(inputs, outputs)
