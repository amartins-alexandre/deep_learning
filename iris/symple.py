import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

base = pd.read_csv('data/iris.csv')
predictors = base.iloc[:, 0:4].values
clazz = base.iloc[:, 4].values

label_encoder = LabelEncoder()
clazz = label_encoder.fit_transform(clazz)
clazz_dummy = to_categorical(clazz)
# setosa        1 0 0
# versicolor    0 1 0
# virginica     0 0 1

predictors_train, predictors_test, target_train, target_test = train_test_split(predictors, clazz_dummy, test_size=0.25)

sequential = Sequential()
sequential.add(Dense(units=4, activation='relu', input_dim=4))
sequential.add(Dense(units=4, activation='relu'))
sequential.add(Dense(units=3, activation='softmax'))
sequential.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
sequential.fit(predictors_train, target_train, batch_size=10, epochs=1000)

result = sequential.evaluate(predictors_test, target_test)
predict = sequential.predict(predictors_test)
predict = (predict > 0.5)

target_converter = [np.argmax(t) for t in target_test]
predict_converter = [np.argmax(t) for t in predict]


matrix = confusion_matrix(predict_converter, target_converter)
