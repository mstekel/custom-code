"""
Example DSW user script.

Fetches train/test data for the current experiment from Pyramid, trains a
classifier, reports evaluation metrics back to Pyramid, and stages the
trained model so it can be reused for scoring.

Intended to run inside a Pyramid Data Science Workspace (DSW) code node.
DswClient reads its connection info (session token, command log id,
callback port, flow type, split number) automatically from CLI args /
the DSW launch configuration - no manual wiring required.
"""

import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from pa.dsw import DswClient


def main():
    client = DswClient()

    # 1. Get data from Pyramid
    data = client.get_data(flow_type="Experiment")

    target = data.target_column
    X_train = data.train_data.drop(columns=[target])
    y_train = data.train_data[target]
    X_test = data.test_data.drop(columns=[target])
    y_test = data.test_data[target]

    # 2. Train a model. Pyramid datasets commonly mix numeric and
    # categorical/text columns, so encode them before handing to sklearn -
    # RandomForestClassifier can't fit on raw strings.
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X_train.select_dtypes(exclude=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols),
    ])

    model = Pipeline([
        ("preprocess", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])

    train_start = time.time()
    model.fit(X_train, y_train)
    training_duration = time.time() - train_start

    # 3. Predict on the held-out test split
    test_start = time.time()
    predicted = model.predict(X_test)
    testing_duration = time.time() - test_start

    predictions = pd.Series(predicted, index=X_test.index)

    # 4. Report results back to Pyramid
    results = client.evaluate_predictions(
        predictions,
        training_duration=training_duration,
        testing_duration=testing_duration,
    )
    print("Evaluation results:", results)

    # 5. Stage the trained model so Pyramid can use it for future scoring
    # client.publish_model(model)


if __name__ == "__main__":
    main()
