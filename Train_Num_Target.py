"""
Pyramid DSW binary-classification example.

Fetches train/test data for the current experiment from Pyramid, validates a
binary target containing 0 and 1, trains a Random Forest classifier, reports
evaluation metrics back to Pyramid, and optionally stages the trained model.

Intended to run inside a Pyramid Data Science Workspace code node.
"""

import time
from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from pa.dsw import DswClient


def prepare_binary_target(
    target_series: pd.Series,
    target_name: str,
) -> pd.Series:
    """
    Validate that the target contains only 0 and 1, then convert the values
    to strings.

    The output classes are:
        0 -> "0"
        1 -> "1"

    String labels help avoid Pyramid's Integer-to-String Java casting error.
    """

    # Convert values such as 0, 1, 0.0 and 1.0 to numeric values.
    numeric_target = pd.to_numeric(target_series, errors="coerce")

    # Detect values that could not be converted to numbers.
    invalid_mask = target_series.notna() & numeric_target.isna()

    if invalid_mask.any():
        invalid_values = (
            target_series.loc[invalid_mask]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            f"Target column '{target_name}' contains non-numeric values: "
            f"{invalid_values}"
        )

    # A classification target cannot contain missing values.
    if numeric_target.isna().any():
        missing_count = int(numeric_target.isna().sum())

        raise ValueError(
            f"Target column '{target_name}' contains "
            f"{missing_count} missing values."
        )

    # Check for values such as 0.5 before converting to integers.
    non_integer_mask = numeric_target % 1 != 0

    if non_integer_mask.any():
        invalid_values = (
            numeric_target.loc[non_integer_mask]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            f"Target column '{target_name}' contains non-integer values: "
            f"{invalid_values}"
        )

    integer_target = numeric_target.astype("int64")

    unique_values = sorted(integer_target.unique().tolist())

    if not set(unique_values).issubset({0, 1}):
        raise ValueError(
            f"Target column '{target_name}' must contain only 0 and 1. "
            f"Found: {unique_values}"
        )

    # Both classes must exist in the training set.
    if len(unique_values) < 2:
        raise ValueError(
            f"Target column '{target_name}' contains only one class: "
            f"{unique_values}. The training data must contain both 0 and 1."
        )

    # Return ordinary Python string labels rather than pandas StringDtype.
    return integer_target.map({0: "0", 1: "1"}).astype(object)


def build_preprocessor(
    X_train: pd.DataFrame,
) -> ColumnTransformer:
    """
    Create preprocessing steps for numeric and categorical predictor columns.
    """

    categorical_columns: List[str] = (
        X_train.select_dtypes(
            include=["object", "category", "string", "bool"]
        )
        .columns
        .tolist()
    )

    numeric_columns: List[str] = [
        column
        for column in X_train.columns
        if column not in categorical_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            )
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError(
            "No predictor columns were found after removing the target."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def main() -> None:
    client = DswClient()

    # ---------------------------------------------------------------
    # 1. Fetch the experiment data from Pyramid
    # ---------------------------------------------------------------

    data = client.get_data(flow_type="Experiment")

    target = data.target_column

    if not target:
        raise ValueError(
            "Pyramid did not provide a target-column name."
        )

    if target not in data.train_data.columns:
        raise ValueError(
            f"Target column '{target}' is missing from the training data."
        )

    if target not in data.test_data.columns:
        raise ValueError(
            f"Target column '{target}' is missing from the test data."
        )

    X_train = data.train_data.drop(columns=[target]).copy()
    X_test = data.test_data.drop(columns=[target]).copy()

    # Convert the numeric 0/1 target into the strings "0" and "1".
    y_train = prepare_binary_target(
        data.train_data[target],
        target,
    )

    # Validate the test target too. It is not passed directly to model.fit(),
    # but validation helps identify inconsistent experiment data.
    y_test = prepare_binary_target(
        data.test_data[target],
        target,
    )

    print(f"Target column: {target}")
    print(
        "Training target counts:",
        y_train.value_counts(dropna=False).to_dict(),
    )
    print(
        "Test target counts:",
        y_test.value_counts(dropna=False).to_dict(),
    )

    # ---------------------------------------------------------------
    # 2. Create the preprocessing and model pipeline
    # ---------------------------------------------------------------

    preprocessor = build_preprocessor(X_train)

    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ]
    )

    # ---------------------------------------------------------------
    # 3. Train the model
    # ---------------------------------------------------------------

    training_start = time.perf_counter()

    model.fit(X_train, y_train)

    training_duration = (
        time.perf_counter() - training_start
    )

    print(
        f"Training completed in "
        f"{training_duration:.4f} seconds."
    )

    # ---------------------------------------------------------------
    # 4. Predict on the test data
    # ---------------------------------------------------------------

    testing_start = time.perf_counter()

    predicted_labels = model.predict(X_test)

    testing_duration = (
        time.perf_counter() - testing_start
    )

    # Explicitly construct predictions as ordinary Python strings.
    # This is intended to prevent Pyramid from receiving Java Integers
    # where its evaluation layer expects String class labels.
    predictions = pd.Series(
        [str(value) for value in predicted_labels],
        index=X_test.index,
        name=target,
        dtype=object,
    )

    invalid_predictions = set(predictions.unique()) - {"0", "1"}

    if invalid_predictions:
        raise ValueError(
            "The model produced unexpected class labels: "
            f"{sorted(invalid_predictions)}"
        )

    print(
        "Prediction counts:",
        predictions.value_counts(dropna=False).to_dict(),
    )

    # ---------------------------------------------------------------
    # 5. Send evaluation results back to Pyramid
    # ---------------------------------------------------------------

    results = client.evaluate_predictions(
        predictions,
        training_duration=training_duration,
        testing_duration=testing_duration,
    )

    print("Evaluation results:", results)

    # ---------------------------------------------------------------
    # 6. Optionally stage the trained model for future scoring
    # ---------------------------------------------------------------

    # Remove the comment when you are ready to publish the model.
    # client.publish_model(model)


if __name__ == "__main__":
    main()
