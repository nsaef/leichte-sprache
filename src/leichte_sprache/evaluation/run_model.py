from argparse import ArgumentParser
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vllm import LLM, SamplingParams

from leichte_sprache.constants import (
    LS_USER_PROMPT_TEXT,
    LS_SYSTEM_PROMPT_DICT,
)
from leichte_sprache.evaluation.score import calculate_metrics
from leichte_sprache.utils.model_utils import generate_vllm
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


def create_prompts(texts: list[str]) -> list[dict]:
    """Create prompts in the chat format from a list of texts.
    In order to evaluate a model finetuned to generate Leichte Sprache,
    use text in standard German as input. The system and user prompt
    used to train the model are used for the generation prompt.

    :param text: list of texts that should be transformed to Leichte Sprache
    :return: list of prompts in the chat format: [{"role": "user", "content": prompt+text}]
    """
    messages = []

    for text in texts:
        message = [
            LS_SYSTEM_PROMPT_DICT,
            {
                "role": "user",
                "content": LS_USER_PROMPT_TEXT.replace("{text_user}", text),
            },
        ]
        messages.append(message)
    return messages


def run_inference_vllm(args: ArgumentParser, prompts: list[str]) -> list[str]:
    """Run inference using vLLM. Generates five examples per prompt.

    :param args: arguments including modelname
    :param prompts: list of prompts in the chat format
    :return: list of generated texts (five per prompt)
    """
    llm = LLM(
        model=args.model_name,
        max_model_len=2048,
    )
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        skip_special_tokens=True,
        top_k=50,
        n=5,
    )
    results = generate_vllm(
        messages=prompts,
        llm=llm,
        sampling_params=sampling_params,
    )
    texts = [output.text for res in results for output in res.outputs]
    return texts


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
    prompts = create_prompts(list(df.text))
    texts = run_inference_vllm(args, prompts)
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
