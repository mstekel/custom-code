"""
Example DSW user script.

Loads the published model, streams incoming data batches from Pyramid,
scores each row, and writes the predictions (with class probabilities)
back to Pyramid.

Intended to run inside a Pyramid Data Science Workspace (DSW) code node.
DswClient reads its connection info (session token, command log id,
callback port, flow type, split number) automatically from CLI args /
the DSW launch configuration - no manual wiring required.
"""

import logging
import sys

_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.addFilter(lambda r: r.levelno < logging.WARNING)
_stderr_handler = logging.StreamHandler(sys.stderr)
_stderr_handler.setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        _stdout_handler,
        _stderr_handler,
    ]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

try:
    from pa.dsw import DswClient, ClassificationPrediction, ClassificationPredictionProbability
except Exception as e:
    logger.error(f"Import failed: {e}", exc_info=True)
    sys.exit(1)


def main():
    client = DswClient()

    model = client.get_model()
    logger.info(f"Model type: {type(model).__name__}")
    if hasattr(model, 'feature_names_in_'):
        logger.info(f"[PREDICT] Model expects features ({len(model.feature_names_in_)}): {list(model.feature_names_in_)}")

    data = client.get_data_stream()

    batch_number = 0
    for batch in data.read():
        batch_number += 1
        target_col = data.target_column
        features = batch.drop(columns=[target_col], errors='ignore') if target_col else batch
        logger.info(f"[PREDICT] Incoming features ({len(features.columns)}): {list(features.columns)}")

        predictions = []
        failed = 0
        for i in features.index:
            try:
                row = features.loc[[i]]
                probabilities = [
                    ClassificationPredictionProbability(label=str(cls), probability=float(prob))
                    for cls, prob in zip(model.classes_, model.predict_proba(row)[0])
                ]
                predictions.append(ClassificationPrediction(
                    id=i,
                    predicted_label=str(model.predict(row)[0]),
                    probabilities=probabilities
                ))
            except Exception as e:
                logger.error(f"Prediction failed for id={i}: {e}", exc_info=True)
                failed += 1

        client.write_predictions(predictions, batch_number)
        logger.info(f"Wrote {len(predictions)} predictions, {failed} failed")

    logger.info("Batch loop complete")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
