import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from scikeras.wrappers import KerasClassifier


base = pd.read_csv('data/iris.csv')
predictors = base.iloc[:, 0:4].values
clazz = base.iloc[:, 4].values

label_encoder = LabelEncoder()
clazz = label_encoder.fit_transform(clazz)
clazz_dummy = to_categorical(clazz)


def create_network():
    sequential = Sequential()
    sequential.add(Dense(units=4, activation='relu', input_dim=4))
    sequential.add(Dense(units=4, activation='relu'))
    sequential.add(Dense(units=3, activation='softmax'))
    sequential.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return sequential


classifier = KerasClassifier(model=create_network, epochs=1000, batch_size=10)
scores = cross_val_score(estimator=classifier, X=predictors, y=clazz_dummy, cv=10, scoring='accuracy')
accuracy = scores.mean()
standard_deviation = scores.std()

