from argparse import ArgumentParser
import os

from datasets import load_dataset
import evaluate
import pandas as pd
from transformers import pipeline

from leichte_sprache.utils.utils import get_logger


logger = get_logger()


def parse_args() -> ArgumentParser:
    """Parse the command line arguments to evaluate a finetuned classifier.

    :return: ArgumentParser with command line arguments.
    """

    parser = ArgumentParser(
        prog="Leichte Sprache classifier evaluation",
        description="Evaluate a finetuned classifier",
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path of the directory containing the checkpoint directories for the finetuned model",
    )
    args = parser.parse_args()
    return args


def evaluate_classifier(model_dir: str):
    """Evaluate the classifier using the evaluation set that was excluded
    from the training. Calculate accuracy, f1, precision and recall, log them
    to the console and save the result as a CSV file in the model's directory.
    The evaluation is run on all checkpoints of the model.

    :param model_dir: training directory containing the checkpoint directories
    """
    metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    dataset = load_dataset(
        os.getenv("HF_CLASSIFICATION_DATASET_NAME"),
        token=os.getenv("HF_TOKEN"),
        split="validation",
    )
    task_evaluator = evaluate.evaluator("text-classification")
    all_results = []

    for dirname in os.listdir(model_dir):
        path = f"{model_dir}/{dirname}"
        logger.info(f"Running evaluation on model {path}")
        pipe = pipeline("text-classification", model=path, device=0)
        results = task_evaluator.compute(
            model_or_pipeline=pipe,
            data=dataset,
            metric=metric,
            label_mapping={"standard_german": 0, "leichte_sprache": 1, "0": 0, "1": 1},
        )
        results["modelname"] = path
        all_results.append(results)
        logger.info(results)

    df = pd.DataFrame(all_results)
    df = df.sort_values(by=["f1", "accuracy", "precision", "recall"], ascending=False)
    logger.info(df)
    df.to_csv(f"{model_dir}/eval.csv")
    return


if __name__ == "__main__":
    args = parse_args()
    evaluate_classifier(args.model_dir)
