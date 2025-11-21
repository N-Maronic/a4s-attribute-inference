import uuid
import numpy as np
import pandas as pd

from a4s_eval.data_model.evaluation import DataShape, Feature, FeatureType, Dataset
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.model_metric_registry import model_metric_registry


def test_attribute_inference():

    X = np.random.randn(20, 2)
    sensitive = (X[:, 0] > 0).astype(int)

    df = pd.DataFrame({
        "f1": X[:, 0],
        "f2": X[:, 1],
        "sensitive": sensitive,
        "label": np.random.randint(0, 2, size=20),
    })

    ds = DataShape(
        features=[
            Feature(pid=uuid.uuid4(), name="f1", feature_type=FeatureType.FLOAT, min_value=None, max_value=None),
            Feature(pid=uuid.uuid4(), name="f2", feature_type=FeatureType.FLOAT, min_value=None, max_value=None),
            Feature(pid=uuid.uuid4(), name="sensitive", feature_type=FeatureType.CATEGORICAL, min_value=None, max_value=None),
        ],
        target=Feature(pid=uuid.uuid4(), name="label", feature_type=FeatureType.CATEGORICAL, min_value=None, max_value=None),
    )

    dataset = Dataset(pid=uuid.uuid4(), shape=ds, data=df)

    class FakeFunctionalModel:
        def predict_proba(self, x):
            logits = np.stack([x[:, 0], -x[:, 0]], axis=1)
            e = np.exp(logits)
            return e / e.sum(axis=1, keepdims=True)

    functional_model = FakeFunctionalModel()

    metric_fn = model_metric_registry.get_functions()["attribute_inference"]
    results = metric_fn(datashape=ds, model=None, dataset=dataset, functional_model=functional_model)

    assert len(results) == 1
    assert isinstance(results[0], Measure)
    assert results[0].name == "attribute_inference_accuracy"
    # FIX: Change 'value' to 'score'
    assert 0 <= results[0].score <= 1