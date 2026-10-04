"""
Example DSW user script.

Trains a classifier on the full dataset for the current experiment and
publishes it to Pyramid so it can be reused for scoring.

Intended to run inside a Pyramid Data Science Workspace (DSW) code node.
DswClient reads its connection info (session token, command log id,
callback port, flow type, split number) automatically from CLI args /
the DSW launch configuration - no manual wiring required.
"""

import logging
import sys

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

    data = client.get_data()

    features = data.train_data.drop(columns=[data.target_column])
    target = data.train_data[data.target_column]

    logger.info(f"[PUBLISH] Target column: {data.target_column}")
    logger.info(f"[PUBLISH] All columns in train_data: {list(data.train_data.columns)}")
    logger.info(f"[PUBLISH] Training features ({len(features.columns)}): {list(features.columns)}")

    model = Pipeline([
        ('enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ('clf', RandomForestClassifier(random_state=42))
    ])
    model.fit(features, target)

    logger.info(f"[PUBLISH] Pipeline trained successfully")

    client.publish_model(model)
    logger.info(f"[PUBLISH] Model published")


if __name__ == "__main__":
    main()
