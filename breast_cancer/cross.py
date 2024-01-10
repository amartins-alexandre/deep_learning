import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import cross_val_score
from scikeras.wrappers import KerasClassifier

inputs = pd.read_csv("data/breast_inputs.csv")
outputs = pd.read_csv("data/breast_output.csv")


def create_network():
    sequential = Sequential()
    sequential.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
    sequential.add(Dropout(rate=0.2))
    sequential.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
    sequential.add(Dropout(rate=0.2))
    sequential.add(Dense(units=1, activation='sigmoid'))
    optimizer = keras.optimizers.Adam(learning_rate=0.001, weight_decay=0.0001)
    sequential.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
    return sequential


classifier = KerasClassifier(model=create_network, epochs=100, batch_size=10)
scores = cross_val_score(estimator=classifier, X=inputs, y=outputs, cv=10)
accuracy = scores.mean()
standard_deviation = scores.std()  # over fitting verification
