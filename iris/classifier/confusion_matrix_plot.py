import os

import matplotlib
import numpy as np

from keras.models import model_from_json
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from classifier.save_neural_network import save_neural_network

iris = load_iris()
filename = 'model/iris.json'

if not os.path.exists(filename):
    save_neural_network()

json_file = open(filename, 'r')
network_structure = json_file.read()
json_file.close()

sequential = model_from_json(network_structure)
sequential.load_weights("model/iris_weights.h5")

predict = sequential.predict(iris.data)
predict = (predict > 0.5)
predict_converter = [np.argmax(t) for t in predict]

target_dummy = to_categorical(iris.target)
target_converter = [np.argmax(t) for t in target_dummy]

cm = confusion_matrix(target_converter, predict_converter, normalize="true")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)

disp.plot(cmap=matplotlib.colormaps["Blues"])

plt.show()
