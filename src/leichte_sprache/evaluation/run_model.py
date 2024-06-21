from argparse import ArgumentParser

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vllm import LLM, SamplingParams

from leichte_sprache.constants import (
    LS_USER_PROMPT_TEXT,
    LS_SYSTEM_PROMPT_DICT,
)
from leichte_sprache.evaluation.score import (
    calculate_readability_scores,
    run_classifier,
)
from leichte_sprache.utils.model_utils import generate_vllm
from leichte_sprache.utils.utils import get_logger


logger = get_logger()


def parse_args() -> ArgumentParser:
    """Parse the command line arguments to select the sources to crawl.

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
    )
    args = parser.parse_args()
    return args


def create_prompts(texts: list[str]) -> list[dict]:
    """Create a prompt in the chat format. If no text input is given,
    use a default example text. In order to evaluate a model finetuned
    to generate Leichte Sprache, use text in standard German as input.
    The system and user prompt used to train the model are used for this
    generation prompt.

    :param text: text that should be transformed to Leichte Sprache. Default: None. In that case, retrieves an example text.
    :return: prompt in the chat format: [{"role": "user", "content": prompt+text}]
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


def run_inference_vllm(args: ArgumentParser, prompts: list[str]):
    """Run inference using VLLM. Currently (vllm==0.4.3) doesn't work properly with peft.
    Check again in a couple of releases.
    """
    # todo: documentation
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
    # todo docstring
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
    # this requires graphics like histograms of logits etc...
    # todo: consider pushing the evaluation to ML Flow? or build a streamlit dashboard?
    # this is too long, the cosole doesn't show the statistics anymore if I run this
    # logger.info("\n\nDetailed Outputs:")
    # for index, row in gen_df.iterrows():
    #     logger.info("\n\n##### New text #####")
    #     logger.info(f"Predicted class: {row.predicted_class}")
    #     logger.info(f"Logits: SG={row.logits_sg}, LS={row.logits_ls}, difference={row.diff_logits}")
    #     logger.info(row.text_gen)
    return


def calculate_metrics(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: list[str],
    orig_texts: list[str],
) -> pd.DataFrame:
    orig_texts_padded = [item for item in orig_texts for _ in range(5)]
    scores = {
        "text_gen": texts,
        "text_orig": orig_texts_padded,
        "predicted_class": [],
        "logits_sg": [],
        "logits_ls": [],
        "flesch_reading_ease": [],
        "wiener_sachtextformel": [],
    }

    for text in texts:
        predicted_class_id, logits = run_classifier(model, tokenizer, text)
        readability = calculate_readability_scores(text)
        scores["predicted_class"].append(predicted_class_id)
        scores["logits_sg"].append(logits[0][0].item())
        scores["logits_ls"].append(logits[0][1].item())
        scores["flesch_reading_ease"].append(readability["flesch_reading_ease"])
        scores["wiener_sachtextformel"].append(readability["wiener_sachtextformel_4"])

    gen_df = pd.DataFrame(scores)
    gen_df["diff_logits"] = abs(gen_df.logits_sg - gen_df.logits_ls)
    return gen_df


def generate_and_evaluate(args):
    # todo docs
    df = pd.read_csv("src/leichte_sprache/dataset/files/test_data_sg.csv")
    prompts = create_prompts(list(df.text))
    texts = run_inference_vllm(args, prompts)
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(args.classification_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.classification_model
    )
    gen_df = calculate_metrics(model, tokenizer, texts, list(df.text))
    print_metrics(gen_df)
    return


if __name__ == "__main__":
    args = parse_args()
    generate_and_evaluate(args)
