import json

import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import LabelEncoder

from classifier.create_network import create_network

data = pd.read_csv("data/iris.csv")
predictors = data.iloc[:, 0:4].values
clazz = data.iloc[:, 4].values

label_encoder = LabelEncoder()
clazz = label_encoder.fit_transform(clazz)
clazz_dummy = to_categorical(clazz)

classifier = KerasClassifier(
    model=create_network,
    hidden_layers_sizes=(4,),
    hidden_layers_activation="relu",
)

parameters = {
    'batch_size': [10, 30, 50],
    'epochs': [1000, 2000, 500],
    'optimizer': ['adam', 'sgd'],
    'loss': ['categorical_crossentropy', 'kl_divergence'],
    'hidden_layers_activation': ['relu', 'tanh'],
    'hidden_layers_sizes': [(8,), (4,), (2,), (1,)]
}
grid_search = GridSearchCV(
    estimator=classifier,
    param_grid=parameters,
    scoring='accuracy',
    cv=10
)
grid_search = grid_search.fit(predictors, clazz)
with open('/model/tuning.json', 'w') as file:
    best_params = json.dumps(grid_search.best_params_, indent=2)
    file.write(best_params)

best_score = grid_search.best_score_
print(best_score)
