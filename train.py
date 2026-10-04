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

import logging
import sys
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from pa.dsw import DswClient

_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.addFilter(lambda r: r.levelno < logging.WARNING)
_stderr_handler = logging.StreamHandler(sys.stderr)
_stderr_handler.setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        _stdout_handler,
        _stderr_handler,
    ]
)
logger = logging.getLogger(__name__)


def main():
    client = DswClient()

    # 1. Get data from Pyramid
    data = client.get_data(flow_type="Experiment")

    target = data.target_column
    X_train = data.train_data.drop(columns=[target])
    y_train = data.train_data[target]
    X_test = data.test_data.drop(columns=[target])
    y_test = data.test_data[target]

    logger.info(f"[TRAIN] Target column: {target}")
    logger.info(f"[TRAIN] Training features ({len(X_train.columns)}): {list(X_train.columns)}")
    logger.info(f"[TRAIN] Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # 2. Train a model
    model = Pipeline([
        ('enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ('clf', RandomForestClassifier(random_state=42))
    ])

    train_start = time.perf_counter()
    model.fit(X_train, y_train)
    training_duration = (time.perf_counter() - train_start) * 1000
    logger.info(f"[TRAIN] Training done in {training_duration:.1f}ms")

    # 3. Predict on the held-out test split
    test_start = time.perf_counter()
    predicted = model.predict(X_test)
    testing_duration = (time.perf_counter() - test_start) * 1000
    logger.info(f"[TRAIN] Inference done in {testing_duration:.1f}ms")

    predictions = pd.Series(predicted, index=X_test.index)

    # 4. Report results back to Pyramid
    results = client.evaluate_predictions(
        predictions,
        training_duration=training_duration,
        testing_duration=testing_duration,
    )
    logger.info(f"[TRAIN] Evaluation submitted: {results}")

    # 5. Stage the trained model so Pyramid can use it for future scoring
    # client.publish_model(model)


if __name__ == "__main__":
    main()
