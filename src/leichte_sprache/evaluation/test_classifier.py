import os

from datasets import load_dataset
import evaluate
import pandas as pd
from transformers import pipeline

from leichte_sprache.utils.utils import get_logger


logger = get_logger()


def evaluate_classifier():
    metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    dataset = load_dataset(
        os.getenv("HF_CLASSIFICATION_DATASET_NAME"),
        token=os.getenv("HF_TOKEN"),
        split="validation",
    )
    task_evaluator = evaluate.evaluator("text-classification")
    all_results = []

    model_dir = "/home/nasrin/data/trainings/roberta-ls-02"
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


if __name__ == "__main__":
    evaluate_classifier()
