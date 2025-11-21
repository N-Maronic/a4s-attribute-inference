# Attribute Inference Metric Implementation

This repository contains the implementation of a **Model Metric** focused on measuring a model's vulnerability to **Attribute Inference Attacks**. 
The development is part of the Cybersecurity and AI course taught at the University of Luxembourg.

## 1\. Repository Information

| Item | Details |
| :--- | :--- |
| **Repository URL** | `https://github.com/N-Maronic/a4s-attribute-inference/` |
| **Branch** | `main` |

-----

## 2\. Metric Details and Assumptions

The implemented metric, called `attribute_inference`, assesses how easy it is for an adversary to infer a **sensitive attribute** of a data subject by simply observing the model's output (e.g., prediction probabilities or predicted classes) on that subject's input. In other words, the metric evaluates how much private information the model is leaking *by accident*.

### Example

Imagine the model predicts if a person will **default on a loan (Yes/No)**. The sensitive attribute is the person's **gender (Male/Female)**, which was used during training but is not an input to the attack.

An **Attribute Inference Attack** works like this:

1.  The attacker feeds a person's data into your loan model and gets the output: "90% probability of **No Default**."
2.  The attacker then feeds this model output ("90% No Default") into a separate, small **attacker model** (like the Logistic Regression implemented in the test).
3.  The attacker model tries to guess the person's **gender** based only on that output probability.

If our `attribute_inference` metric reports an accuracy of **95%**, it means your model's prediction is leaking enough information that an attacker can correctly guess the sensitive attribute 95% of the time, which is a serious privacy regulation.

### Metric Name and Location

  * **Metric Name:** `attribute_inference`
  * **Implementation File:** `a4s_eval/metrics/model_metrics/attribute_inference.py`
  * **Test File:** `tests/metrics/model_metrics/test_attribute_inference.py`

### Key Assumptions and Applicability

This metric is a **Model Metric**, meaning it evaluates a deployed model's behavior using a given dataset. It makes the following key assumptions:

| Assumption | Description |
| :--- | :--- |
| **Model Type** | Applicable to **classification models** (binary or multi-class). |
| **Model Output** | The `functional_model` must expose either: \* `predict_proba(X)` (preferred, returns probability scores) OR \* `predict_class(X)` (returns hard predicted labels). |
| **Data Requirement** | Requires a **labeled dataset** (`Dataset.data`) that includes the **sensitive attribute**. |
| **Sensitive Attribute** | The sensitive attribute must be **categorical** (typically binary, e.g., 0 or 1). |
| **Attacker Model** | The metric uses a **Logistic Regression** classifier as the standard attribute inference attacker. |

### Metric Mechanism

The attack proceeds in these steps:

1.  The target model's predictions (`outputs`) on the input data $X$ are collected.
2.  A **Logistic Regression** classifier (the attacker) is trained to predict the **sensitive attribute** (`sensitive`) using only the target model's `outputs` as features.
3.  The attacker's success is measured by its **prediction accuracy** on the `outputs`.
    $$\text{Accuracy} = \frac{\text{Number of correct predictions of sensitive attribute}}{\text{Total number of data points}}$$

-----

## 3\. Test Implementation and Execution

The functionality of the `attribute_inference` metric is verified using an included pytest.

### Test Details

  * **Test Function:** `test_attribute_inference()`
  * **Test File:** `tests/metrics/model_metrics/test_attribute_inference.py`
  * **Test Goal:** Verify that the metric function runs without error and returns a valid `Measure` object with an accuracy score between 0 and 1.

### Test Components

| Component | Description |
| :--- | :--- |
| **Data** | A synthetic `pandas.DataFrame` (`df`) with 20 samples and 4 features (`f1`, `f2`, `sensitive`, `label`). |
| **Sensitive Attribute** | Created artificially based on the first feature: `sensitive = (X[:, 0] > 0).astype(int)`. This makes the sensitive attribute *correlated* with the input feature `f1`. |
| **Functional Model** | A fake class, `FakeFunctionalModel`, whose `predict_proba` function is designed to leak information about the sensitive feature: `logits = np.stack([x[:, 0], -x[:, 0]], axis=1)`. Since the sensitive attribute is based on $x[:, 0]$, the model output is directly related to the sensitive attribute, ensuring the resulting accuracy score is typically high (near 1.0) and the test is meaningful. |
| **Expected Measure** | Returns one measure: `attribute_inference_accuracy`. |

## 4. Next Steps 

The implemented metric is currently verified using a minimal, synthetic test case. For the next phase, I will focus on proper real-world validation and integration into a robust privacy auditing pipeline, drawing inspiration from the Privacy Meter library's methodology.

### 4.1. Testing with proper Benchmark Datasets

The primary goal is to demonstrate the metric's efficacy on established benchmark datasets known for privacy risk evaluation:

- Datasets: I'm planning to use the Purchase100 and Texas100 datasets, which are standard for evaluating privacy defenses against Membership and Attribute Inference Attacks.

### 4.2. Implementation of the Test Scenario

1. Target Model Training: A suitable target model will be trained on the full dataset, including all input features and the primary prediction label.

2. Sensitive Attribute Selection: One feature will be designated as the sensitive attribute (e.g., Race or Gender from the Texas100 dataset).

3. Attacker Training and Evaluation:

    - The sensitive attribute will be removed from the features passed to the functional_model for prediction.

    - The attribute_inference metric will be executed. It will take the target model's output (prediction probabilities) and train the internal LogisticRegression attacker to predict the removed sensitive attribute.

4. Goal: Achieve a quantitative measure of privacy leakage (Attribute Inference Accuracy) under a realistic threat model where the attacker only observes the model's output.
