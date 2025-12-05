# Attribute Inference Metric Implementation

This repository contains the implementation of a **Model Metric** focused on measuring a model's vulnerability to **Attribute Inference Attacks**. 
The development is part of the Cybersecurity and AI course taught at the University of Luxembourg.

## 1\. Repository Information and Installation

| Item | Details |
| :--- | :--- |
| **Repository URL** | `https://github.com/N-Maronic/a4s-attribute-inference/` |
| **Branch** | `main` |


To install all necessary components for the Attribute Inference metric into your existing **a4s-eval** framework, simply run the provided `install.py` script. The script will automatically download the Adult Census dataset, place it in the correct directory, install the metric implementation, test file, demo notebook, and update the project configuration. When prompted, supply the **absolute path to the root of your a4s-eval repository** (the folder containing both `a4s_eval/` and `tests/`). After completion, your environment will be fully configured and ready to run the metric and accompanying demo.

-----

## 2\. Metric Details and Assumptions

The implemented metric, called `attribute_inference`, assesses how easy it is for an adversary to infer a **sensitive attribute** of a data subject by simply observing the model's output (e.g., prediction probabilities or predicted classes) on that subject's input. In other words, the metric evaluates how much private information the model is leaking *by accident*.

### Example

Imagine the model predicts if a person will **default on a loan (Yes/No)**. The sensitive attribute is the person's **gender (Male/Female)**, which was used during training but is not an input to the attack.

An **Attribute Inference Attack** works like this:

1.  The attacker feeds a person's data into your loan model and gets the output: "90% probability of **No Default**."
2.  The attacker then feeds this model output ("90% No Default") into a separate, small **attacker model** (like the Logistic Regression implemented in the test).
3.  The attacker model tries to guess the person's **gender** based only on that output probability.

If our `attribute_inference` metric reports an accuracy of **95%**, it means your model's prediction is leaking enough information that an attacker can correctly guess the sensitive attribute 95% of the time, which is a serious privacy violation.

### Metric Name and Location

  * **Metric Name:** `attribute_inference`
  * **Implementation File:** `a4s_eval/metrics/model_metrics/attribute_inference.py`
  * **Test File:** `tests/metrics/model_metrics/test_attribute_inference.py`

### Key Assumptions and Applicability

This metric is a **Model Metric**, meaning it evaluates a deployed model's behavior using a given dataset. It makes the following key assumptions:

| Assumption | Description |
| :--- | :--- |
| **Model Type** | Applicable to **classification models** (binary or multi-class). |
| **Model Output** | The `functional_model` must expose either: `predict_proba(X)` (preferred, returns probability scores) OR `predict_class(X)` (returns hard predicted labels). |
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


## 4\. Notebook Demo: End-to-End Attribute Inference Evaluation

This repository includes a complete **Jupyter Notebook demo** showing how to use the implemented `attribute_inference` metric in a realistic privacy evaluation scenario.
The notebook walks through the entire workflow: loading a dataset, preprocessing, training models, running attribute inference attacks, and interpreting the results.

**Notebook File:**

```
attribute_inference_demo.ipynb
```



### **1. Loading the Adult Census Dataset**

The notebook uses the **Adult Census Income** dataset from Kaggle (`data/adult.csv`).
This dataset is a standard benchmark in fairness and privacy research because it contains:

* meaningful sensitive attributes (`race`, `sex`)
* rich demographic information
* realistic correlations between features
* a real prediction task (income > 50K)

These characteristics make it ideal for demonstrating attribute inference attacks.

---

### **2. Preprocessing and Feature Encoding**

The Adult dataset contains both numeric and categorical variables.
To make it usable for machine learning models, all categorical features are transformed using **one-hot encoding**.

Encoding is essential because:

* machine learning models require numerical input
* correlation analysis on strings is not meaningful
* subtle demographic patterns become detectable to both the victim model and the attacker
* removing a sensitive attribute does *not* eliminate demographic information due to redundant encodings (e.g., race correlates with native-country)

---

### **3. Building A4S-Compatible Functional Wrappers**

The A4S framework expects a standardized model interface:

```python
predict_proba(x) -> np.ndarray
```

To meet this requirement, the notebook wraps each trained model (Logistic Regression and Neural Network) inside a **functional model wrapper** that:

* applies the correct preprocessing (e.g., scaling)
* ensures the correct feature order
* removes sensitive features for the “without attribute” models
* presents a clean API to the A4S `attribute_inference` metric


---

### **4. Training Victim Models**

Two commonly used classification models are trained:

* **Logistic Regression (LR)** — interpretable, stable, and widely used in fairness/privacy work
* **Neural Network (NN)** — higher capacity, capable of capturing nonlinear patterns

The notebook trains:

* a **baseline model** using all features
* a **modified model** where one sensitive attribute (e.g., race) is removed


---

### **5. Running Attribute Inference Attacks**

For each sensitive attribute (race, sex, education), the notebook evaluates two scenarios:

* **With attribute:** victim model trained using all features
* **Without attribute:** victim model trained without the sensitive feature(s)

The A4S `attribute_inference` metric trains an attacker model that tries to predict the sensitive attribute *only* from the victim model’s output probabilities.

The result is an **attribute inference accuracy** for each setting and model type.

---

### **6. Interpreting the Results**

The notebook includes a dedicated interpretation section explaining:

* why **race inference** is very high (strong multivariate signal despite low Pearson correlation)
* why **removing the sensitive attribute does not reduce leakage** (redundancy in correlated features)
* why **LR and NN behave similarly** (similar output distributions, attacker only sees probabilities)
