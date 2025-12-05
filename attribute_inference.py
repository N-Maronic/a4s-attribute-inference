from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from a4s_eval.metric_registries.model_metric_registry import model_metric
from a4s_eval.data_model.measure import Measure


@model_metric(name="attribute_inference")
def attribute_inference_metric(
    datashape,
    model,
    dataset,
    functional_model,
    sensitive_column: str | None = None,
) -> list[Measure]:

    # ---------------------------
    # Load dataframe
    # ---------------------------
    df = dataset.data
    if df is None:
        raise ValueError("Attribute inference metric requires dataset.data to be a pandas DataFrame.")

    # ---------------------------
    # Identify sensitive attribute
    # ---------------------------
    if sensitive_column is not None:
        if sensitive_column not in df.columns:
            raise ValueError(f"Sensitive column '{sensitive_column}' not found in dataset.")
        sensitive = df[sensitive_column].to_numpy()
    elif "sensitive" in df.columns:
        sensitive = df["sensitive"].to_numpy()
    else:
        # fallback heuristic (for backwards compatibility or simple datasets)
        non_target_features = [f.name for f in datashape.features]
        target_name = datashape.target.name if datashape.target else None
        candidate_features = [c for c in df.columns if c in non_target_features and c != target_name]

        if len(candidate_features) == 0:
            raise ValueError(
                "Could not identify sensitive attribute. "
                "Specify sensitive_column explicitly."
            )

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
    # Train/Test split for attacker
    # ---------------------------
    # try stratification, fallback to non-stratified
    try:
        splits = train_test_split(
            outputs, sensitive,
            test_size=0.3,
            random_state=42,
            stratify=sensitive
        )
    except ValueError:
        splits = train_test_split(
            outputs, sensitive,
            test_size=0.3,
            random_state=42,
            stratify=None
        )

    outputs_train, outputs_test, sensitive_train, sensitive_test = splits


    # ---------------------------
    # Train attacker model
    # ---------------------------
    attacker = LogisticRegression(max_iter=500)
    attacker.fit(outputs_train, sensitive_train)

    # ---------------------------
    # Evaluate attacker
    # ---------------------------
    pred = attacker.predict(outputs_test)
    acc = float((pred == sensitive_test).mean())

    return [
        Measure(
            name="attribute_inference_accuracy",
            score=acc,
            time=0.0,
        )
    ]
