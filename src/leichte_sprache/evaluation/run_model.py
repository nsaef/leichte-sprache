from argparse import ArgumentParser

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vllm import LLM, SamplingParams

from leichte_sprache.constants import (
    TEST_ARTICLE,
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


def create_prompt(text: str = None) -> list[dict]:
    """Create a prompt in the chat format. If no text input is given,
    use a default example text. In order to evaluate a model finetuned
    to generate Leichte Sprache, use text in standard German as input.
    The system and user prompt used to train the model are used for this
    generation prompt.

    :param text: text that should be transformed to Leichte Sprache. Default: None. In that case, retrieves an example text.
    :return: prompt in the chat format: [{"role": "user", "content": prompt+text}]
    """
    if not text:
        text = TEST_ARTICLE

    messages = []
    message = [
        LS_SYSTEM_PROMPT_DICT,
        {
            "role": "user",
            "content": LS_USER_PROMPT_TEXT.replace("{text_user}", text),
        },
    ]
    messages.append(message)
    return messages


def calculate_scores(texts: list[str]) -> pd.DataFrame:
    # todo docs
    scores = [calculate_readability_scores(text) for text in texts]
    df = pd.DataFrame(scores)
    logger.info(df.describe().loc["mean"])
    return df


def run_inference_vllm(args: ArgumentParser):
    """Run inference using VLLM. Currently (vllm==0.4.3) doesn't work properly with peft.
    Check again in a couple of releases.
    """
    # todo: documentation

    messages = create_prompt()

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
        messages=messages,
        llm=llm,
        sampling_params=sampling_params,
    )
    texts = [output.text for res in results for output in res.outputs]
    return texts


def generate_and_evaluate(args):
    # todo docs
    texts = run_inference_vllm(args)
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(args.classification_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.classification_model
    )
    for text in texts:
        logger.info("\n\n #-#-#-#-#-#-#-#-#")
        try:
            predicted_class_id, logits = run_classifier(model, tokenizer, text)
            logger.info(
                f"Predicted class ID: {predicted_class_id} (0=SG, 1=LS); Logits: {logits}"
            )
        except Exception as e:
            logger.debug(f"Encountered exception: {e}\nSkipping classification.")
        logger.info(text)


if __name__ == "__main__":
    args = parse_args()
    generate_and_evaluate(args)
