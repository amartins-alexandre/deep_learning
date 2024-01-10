import json

from keras.utils import to_categorical
from sklearn.datasets import load_iris

from classifier.create_network import create_network


def save_neural_network():
    iris = load_iris()
    clazz_dummy = to_categorical(iris.target)

    with open('model/tuning.json') as json_file:
        best_params = json.load(json_file)
        model = create_network(
            meta={'n_classes_': len(iris.target_names),
                  'n_features_in_': len(iris.data[0]),
                  'target_type_': 'multiclass'},
            compile_kwargs={'loss': best_params['loss'], 'optimizer': best_params['optimizer']},
            hidden_layers_sizes=best_params['hidden_layers_sizes'],
            hidden_layers_activation=best_params['hidden_layers_activation'],
        )
        model.fit(iris.data,
                  clazz_dummy,
                  batch_size=best_params['batch_size'],
                  epochs=best_params['epochs'])

        data_json = model.to_json()
        with open("model/iris.json", "w") as json_file:
            json_file.write(data_json)

        model.save_weights("model/iris_weights.h5")
