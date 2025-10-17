from datetime import datetime
import numpy as np
import pandas as pd

from a4s_eval.data_model.evaluation import DataShape, Dataset, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.model_metric_registry import model_metric
from a4s_eval.service.model_functional import FunctionalModel


@model_metric(name="accuracy")
def accuracy(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    functional_model: FunctionalModel,
) -> list[Measure]:
    
    # Both x and y (the features and the target) are contained in dataset.data as a dataframe.
    data = dataset.data

    # To identify the target (y), use the datashape.target object, which has a name property. Use this property to index the aforementioned dataframe.
    target_name = datashape.target.name
    y_true = data[target_name]

    # To identify the features (x), use the datashape.features list of object. Similarly each object in this list has a name property to index the dataframe.
    feature_names = [f.name for f in datashape.features]
    X = data[feature_names]

    # Inspect FunctionalModel definition to identify the function to use to compute the model predictions.
    # -> in a4s_eval/service/model_functional.py there is def predict(x: Array) -> Array:
    y_pred = functional_model.predict(X.to_numpy())

    # Use the y (from the dataset.data) and the prediction to cumpute the accuracy.
    
    # Below is a placeholder that allows pytest to pass.
    
    # If this takes too many resources (e.g., runs very long or causes a memory error), feel free to limit the dataset to the first 10,000 examples.
    if len(X) > 10000:
        X = X.iloc[:10000]
        y_true = y_true.iloc[:10000]

    # Convert to numpy array
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)

    # Compute accuracy
    correct = np.sum(y_true == y_pred)
    accuracy_value = correct / len(y_true) if len(y_true) > 0 else 0.0
    
    current_time = datetime.now()
    return [Measure(name="accuracy", score=accuracy_value, time=current_time)]
