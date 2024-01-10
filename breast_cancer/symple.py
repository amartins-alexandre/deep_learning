import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import keras
from keras.models import Sequential
from keras.layers import Dense


inputs = pd.read_csv("data/breast_inputs.csv")
outputs = pd.read_csv("data/breast_output.csv")

predictor_train, predictor_test, classifier_train, classifier_test = train_test_split(inputs, outputs,
                                                                                      test_size=0.25)

# Construction Neural Network
classifier = Sequential()
classifier.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
classifier.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
classifier.add(Dense(units=1, activation='sigmoid'))

optimizer = keras.optimizers.Adam(learning_rate=0.001, decay=0.0001)
# classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

# Training
classifier.fit(predictor_train, classifier_train, batch_size=10, epochs=100)

# Weights View
weights = classifier.layers[0].get_weights()

# Validation
predict = classifier.predict(predictor_test)
predict = (predict > 0.5)
pressure = accuracy_score(classifier_test, predict)
cm = confusion_matrix(classifier_test, predict)
result = classifier.evaluate(predictor_test, classifier_test)

