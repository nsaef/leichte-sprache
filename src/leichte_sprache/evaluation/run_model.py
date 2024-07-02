from argparse import ArgumentParser
import os

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from leichte_sprache.evaluation.score import calculate_metrics
from leichte_sprache.inference.inference import run_inference_vllm
from leichte_sprache.utils.model_utils import create_prompts_sg_to_ls
from leichte_sprache.utils.utils import get_logger


logger = get_logger()


def parse_args() -> ArgumentParser:
    """Parse the command line arguments to run and evaluate a finetuned model.

    :return: ArgumentParser with command line arguments.
    """

    parser = ArgumentParser(
        prog="Leichte Sprache Inference",
        description="Run inference on a finetuned model",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="Name or path of the base model of the finetuned model",
    )
    parser.add_argument(
        "--classification_model",
        help="Name or path of a classifier for Leichte Sprache",
        required=True,
    )
    args = parser.parse_args()
    return args


def print_metrics(df: pd.DataFrame):
    """Analyse the calculated metrics and log summaries to the console.

    :param df: dataframe containing metrics from the classifier and readability scores
    """
    share_sg = len(df[df.predicted_class == 0]) / len(df.dropna())
    logger.info(df)
    logger.info(f"Mean predicted class: {df.dropna().predicted_class.mean()}")
    logger.info(
        f"Share of texts not classified as Leichte Sprache: {round(share_sg*100, 2)} %"
    )
    logger.info(f"Mean difference between logits: {df.diff_logits.mean()}")
    logger.info(
        f"Mean logits of LS texts: {df[df.predicted_class == 1].logits_ls.mean()}"
    )
    logger.info(
        f"Min logits of LS texts: {df[df.predicted_class == 1].logits_ls.min()}"
    )
    logger.info(
        f"Max logits of LS texts: {df[df.predicted_class == 1].logits_ls.max()}"
    )
    logger.info(f"Mean Flesch Reading Ease score: {df.flesch_reading_ease.mean()}")
    logger.info(
        f"Mean Flesch Reading Ease score for LS: {df[df.predicted_class == 1].flesch_reading_ease.mean()}"
    )
    logger.info(f"Mean Wiener Sachtextformel score: {df.wiener_sachtextformel.mean()}")
    logger.info(
        f"Mean Wiener Sachtextformel score for LS: {df[df.predicted_class == 1].wiener_sachtextformel.mean()}"
    )
    return


def generate_and_evaluate(args):
    """Generate Leichte Sprache with a finetuned model based on example data.
    For each row in the example dataset, five samples are generated.
    Calculate metrics on the generated texts using the Leichte Sprache classifier
    and readability metrics and log the results to the console.

    :param args: CLI arguments
    """
    df = pd.read_csv("src/leichte_sprache/dataset/files/test_data_sg.csv")
    prompts = create_prompts_sg_to_ls(list(df.text))
    texts = run_inference_vllm(args.model_name, prompts)
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(args.classification_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.classification_model
    )
    orig_texts = [item for item in list(df.text) for _ in range(5)]
    gen_df = calculate_metrics(model, tokenizer, texts, original_texts=orig_texts)

    print_metrics(gen_df)
    if os.path.exists(args.model_name):
        gen_df.to_csv(f"{args.model_name}/metrics.csv")
    return


if __name__ == "__main__":
    args = parse_args()
    generate_and_evaluate(args)
