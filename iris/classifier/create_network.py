from typing import Dict, Iterable, Any
from keras import Sequential
from keras.layers import Dense, Input


def create_network(
        meta: Dict[str, Any],
        compile_kwargs: Dict[str, Any],
        hidden_layers_activation: str,
        hidden_layers_sizes: Iterable[int]
):
    model = Sequential()

    # Input Layer
    input_layer = Input(shape=(meta['n_features_in_']))
    model.add(input_layer)

    # Hidden Layers
    for size in hidden_layers_sizes:
        model.add(Dense(units=size, activation=hidden_layers_activation))

    # Output Layer
    if meta["target_type_"] == "binary":
        n_output_units = 1
        output_activation = "sigmoid"
    elif meta["target_type_"] == "multiclass":
        n_output_units = meta["n_classes_"]
        output_activation = "softmax"
    else:
        raise NotImplementedError(f"Unsupported task type: {meta['target_type_']}")
    model.add(Dense(n_output_units, activation=output_activation))

    model.compile(loss=compile_kwargs['loss'], optimizer=compile_kwargs['optimizer'])
    return model
