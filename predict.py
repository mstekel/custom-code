"""
Example DSW user script (vectorised version).

Loads the published model, streams incoming data chunks from Pyramid, scores each
chunk with ONE model call per method (instead of one call per row), and writes the
predictions (with class probabilities) back to Pyramid.

Why: scikit-learn pays a fixed validation/dispatch cost per call, so calling
predict_proba() + predict() once per row made the Python process CPU-bound at
~200 rows/s. Scoring the whole chunk at once amortises that cost over every row.

Behaviour is unchanged: same predicted labels (model.predict is still the source of
the label), same probabilities, same prediction ids, and a row that cannot be scored
is still reported by id and skipped. If a chunk fails as a whole, it is retried row
by row so one bad row does not lose the other rows of the chunk.

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
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        _stdout_handler,
        _stderr_handler,
    ]
)

logger = logging.getLogger(__name__)

try:
    from pa.dsw import DswClient, ClassificationPrediction, ClassificationPredictionProbability
except Exception as e:
    logger.error(f"Import failed: {e}", exc_info=True)
    sys.exit(1)


def score_chunk(model, class_labels, values, ids):
    """Scores a whole chunk with a single predict_proba and a single predict call."""
    probabilities = model.predict_proba(values)
    labels = model.predict(values)
    return [
        ClassificationPrediction(
            id=row_id,
            predicted_label=str(label),
            probabilities=[
                ClassificationPredictionProbability(label=cls, probability=float(prob))
                for cls, prob in zip(class_labels, row_probs)
            ]
        )
        for row_id, label, row_probs in zip(ids, labels, probabilities)
    ]


def score_rows_individually(model, class_labels, values):
    """Fallback when scoring a chunk as a whole fails: score row by row so only the bad rows are lost."""
    predictions = []
    failed = 0
    for i in values.index:
        try:
            predictions.extend(score_chunk(model, class_labels, values.loc[[i]], [i]))
        except Exception as e:
            logger.error(f"Prediction failed for id={i}: {e}", exc_info=True)
            failed += 1
    return predictions, failed


def main():
    client = DswClient()

    model = client.get_model()
    logger.info(f"Model type: {type(model).__name__}")
    if hasattr(model, 'feature_names_in_'):
        logger.info(f"[PREDICT] Model expects features ({len(model.feature_names_in_)}): {list(model.feature_names_in_)}")

    class_labels = [str(cls) for cls in model.classes_]

    data = client.get_data_stream()

    batch_number = 0
    for batch in data.read():
        batch_number += 1
        target_col = data.target_column
        values = batch.drop(columns=[target_col], errors='ignore') if target_col else batch
        if batch_number == 1:
            logger.info(f"[PREDICT] Incoming values ({len(values.columns)}): {list(values.columns)}")

        failed = 0
        try:
            predictions = score_chunk(model, class_labels, values, list(values.index))
        except Exception as e:
            logger.warning(f"Chunk {batch_number} failed as a whole ({e}); retrying row by row")
            predictions, failed = score_rows_individually(model, class_labels, values)

        client.write_predictions(predictions, batch_number)
        logger.info(f"Batch {batch_number}: wrote {len(predictions)} predictions, {failed} failed")

    logger.info("Batch loop complete")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
