import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier


inputs = pd.read_csv("data/breast_inputs.csv")
outputs = pd.read_csv("data/breast_output.csv")


def create_network(compile_kwargs, activation):
    sequential = Sequential()
    sequential.add(Dense(units=8, activation=activation, kernel_initializer='random_uniform', input_dim=30))
    sequential.add(Dropout(rate=0.2))
    sequential.add(Dense(units=8, activation=activation, kernel_initializer='random_uniform'))
    sequential.add(Dropout(rate=0.2))
    sequential.add(Dense(units=1, activation='sigmoid'))
    # optimizer = keras.optimizers.Adam(learning_rate=0.001, weight_decay=0.0001)
    sequential.compile(optimizer=compile_kwargs['optimizer'], loss=compile_kwargs['loss'], metrics=['binary_accuracy'])
    return sequential


classifier = KerasClassifier(model=create_network, activation='relu')
parameters = {
    'batch_size': [10, 30],
    'epochs': [50, 100],
    'optimizer': ['adam', 'sgd'],
    'loss': ['binary_crossentropy', 'hinge'],
    'activation': ['relu', 'tanh']
}
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=5)
grid_search = grid_search.fit(inputs, outputs)
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# scores = cross_val_score(estimator=classifier, X=inputs, y=outputs, cv=10, scoring='accuracy')
# accuracy = scores.mean()
# standard_deviation = scores.std()  # over fitting verification
