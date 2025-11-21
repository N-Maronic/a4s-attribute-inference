from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression

from a4s_eval.metric_registries.model_metric_registry import model_metric
from a4s_eval.data_model.measure import Measure


@model_metric(name="attribute_inference")
def attribute_inference_metric(
    datashape,
    model,
    dataset,
    functional_model,
) -> list[Measure]:

    df = dataset.data
    if df is None:
        raise ValueError("Attribute inference metric requires a dataframe (dataset.data).")

    # ---------------------------
    # Find sensitive attribute
    # ---------------------------
    if "sensitive" in df.columns:
        sensitive = df["sensitive"].to_numpy()
    else:
        # use last non-target feature as fallback
        non_target_features = [f.name for f in datashape.features]
        target_name = datashape.target.name if datashape.target else None

        candidate_features = [
            c for c in df.columns if c in non_target_features and c != target_name
        ]

        if len(candidate_features) == 0:
            raise ValueError("Could not find sensitive attribute in dataset.")

        sensitive = df[candidate_features[-1]].to_numpy()

    sensitive = sensitive.astype(int)
    
    # ---------------------------
    # Build input X  
    # ---------------------------
    feature_names = [f.name for f in datashape.features if f.name in df.columns]
    X = df[feature_names].to_numpy()

    # ---------------------------
    # Compute model outputs
    # ---------------------------
    if hasattr(functional_model, "predict_proba") and callable(functional_model.predict_proba):
        outputs = np.asarray(functional_model.predict_proba(X))
    elif hasattr(functional_model, "predict_class") and callable(functional_model.predict_class):
        outputs = np.asarray(functional_model.predict_class(X)).reshape(-1, 1)
    else:
        raise RuntimeError("Functional model has neither predict_proba nor predict_class")

    # ---------------------------
    # Train attribute inference attacker
    # ---------------------------
    attacker = LogisticRegression(max_iter=500)
    attacker.fit(outputs, sensitive)

    pred = attacker.predict(outputs)
    acc = float((pred == sensitive).mean())

    return [Measure(
        name="attribute_inference_accuracy", 
        score=acc, 
        time=0.0    # placeholder
    )]