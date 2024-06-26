from argparse import ArgumentParser
from hashlib import md5
import re

from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams

from leichte_sprache.constants import (
    PROMPTS_COLUMN,
    LS_SYSTEM_PROMPT_DICT,
    LS_USER_PROMPT_TEXT,
)
from leichte_sprache.dataset.data_management import push_to_hf_hub
from leichte_sprache.evaluation.run_model import create_prompts, calculate_metrics
from leichte_sprache.utils.db_utils import ingest_pandas, get_connector
from leichte_sprache.utils.model_utils import generate_vllm, run_vllm_batch_generation
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
    parser.add_argument(
        "--dataset_target_size",
        help="Target size of the dataset to create. Expect the actual dataset to be about 66% of this number, as about one third of rows is filtered due to length contstraints.",
        default=1000,
    )
    args = parser.parse_args()
    return args


def get_domain_allowlist() -> list[str]:
    allowlist = [
        "www.presseportal.de",
        "www.welt.de",
        "www.tagesspiegel.de",
        "derstandard.at",
        "www.handelsblatt.com",
        "www.t-online.de",
        "rp-online.de",
        "www.krone.at",
        "www.spiegel.de",
        "www.sueddeutsche.de",
        "www.ots.at",
        "www.focus.de",
        "www.zeit.de",
        "www.rp-online.de",
        "www.augsburger-allgemeine.de",
        "www.dw.com",
        "www.spox.com",
        "www.stern.de",
        "www.faz.net",
        "www.wz.de",
        "www.oe24.at",
        "www.derstandard.at",
        "www.wiwo.de",
        "www.derwesten.de",
        "www.hna.de",
        "www.shz.de",
        "www.merkur.de",
        "www.heise.de",
        "www.saarbruecker-zeitung.de",
        "taz.de",
        "www.aachener-zeitung.de",
        "www.op-marburg.de",
        "www.weser-kurier.de",
        "www.fr.de",
        "www.ostsee-zeitung.de",
        "www.golem.de",
    ]
    return allowlist


def get_wiki_dataset():
    wiki_dataset = (
        load_dataset("PatrickHaller/wikitext-18-de", split="train").shuffle()
        # .take(1000)
        # .map(lambda x: {"content": f"{x['title']}\n{x['text']}"})
    )
    # split the texts into much smaller units
    articles = []
    urls = []
    rx = r"=+ [\w\s]+ =+"
    for article in tqdm(wiki_dataset, desc="Processing wiki dataset"):
        sections = re.split(rx, article["text"])
        section_texts = [s.strip() for s in sections if s.strip()]
        articles.extend(section_texts)
        urls.extend([article["url"]] * len(section_texts))
    # return split dataset
    wiki_ds = Dataset.from_dict(
        {
            "content": articles,
            "source": ["wiki" for _ in range(len(articles))],
            "url": urls,
        }
    )
    return wiki_ds


def get_web_dataset(target_size: int = 3000):
    web_dataset = load_dataset(
        "uonlp/CulturaX", "de", split="train", streaming=True
    ).shuffle()
    # iterate through dataset
    allowlist = get_domain_allowlist()
    rx = r"https?:\/\/([\w.-]+)\/"
    articles = []

    with tqdm(total=target_size, desc="Processing web dataset") as pbar:
        for article in web_dataset:
            match = re.findall(rx, article["url"])
            if match:
                if match[0] in allowlist:
                    articles.append(article)
                    pbar.update(1)
                if len(articles) == target_size:
                    break
    web_ds = Dataset.from_list(articles).rename_column("text", "content")
    return web_ds


def get_news_dataset():
    # todo docstring
    # load a news dataset, as this is the most common text type in the LS dataset
    news_dataset = (
        load_dataset("bjoernp/tagesschau-010124-020524", split="train")
        .map(
            lambda x: {
                "content": f"{x['headline']}\n{x['article']}",
                "source": "tagesschau",
            }
        )
        .rename_column("link", "url")
        .shuffle()
    )
    return news_dataset


def build_standard_german_dataset(tokenizer_path: str, target_size: int) -> Dataset:
    # todo docstring
    size_per_dataset = round(target_size / 3)

    wiki_ds = get_wiki_dataset()
    web_ds = get_web_dataset(target_size=size_per_dataset)
    news_ds = get_news_dataset()
    valid_columns = ["content", "source", "url"]
    dataset = concatenate_datasets(
        [
            wiki_ds.shuffle()
            .select(range(size_per_dataset))
            .select_columns(valid_columns),
            web_ds.select_columns(valid_columns),
            news_ds.select_columns(valid_columns).select(range(size_per_dataset)),
        ]
    ).shuffle()

    # filter by length to avoid getting lots of texts that are cut short
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokens_prompt = len(tokenizer(LS_USER_PROMPT_TEXT)["input_ids"]) + len(
        tokenizer(LS_SYSTEM_PROMPT_DICT["content"])["input_ids"]
    )
    text_len = calculate_usable_text_length(
        tokenizer=tokenizer,
        texts=dataset["content"],
        prompt_len=tokens_prompt,
        model_max_len=2048,
    )
    dataset = dataset.add_column("usable", text_len)
    dataset = dataset.filter(lambda x: x["usable"] is True)
    return dataset


def calculate_usable_text_length(
    tokenizer: AutoTokenizer, texts: list[str], prompt_len: int, model_max_len: int
):
    # todo docstring
    res = []
    space_for_text = model_max_len - prompt_len

    for text in tqdm(texts, desc="Calculating text length"):
        len_naive = len(text.split())
        # one word usually consists of multiple tokens, especially in non-English languages
        #  if the naively split text cannot be fit into the model twice (input & output), consider it too long
        if len_naive > space_for_text / 2:
            res.append(False)
        else:
            tokens = tokenizer(text)
            n_tokens = len(tokens["input_ids"])
            # assume that the output text is shorter than the input, so leave a bit of room
            # discard very short texts
            if space_for_text - n_tokens < n_tokens * 0.5 or n_tokens < 50:
                res.append(False)
            else:
                res.append(True)
    return res


def run_vllm_generation(
    dataset: Dataset,
    model_id: str,
    table_name: str,
    prompt_col_name: str,
    result_col_name: str,
    batch_size: int = 10,
    n_sequences: int = 5,
):
    """Generate output from prompts in a dataset using VLLM. The prompts are batched
     according the the `batch_size` parameter. The outputs of each batch are stored
     in a database.
     #todo docstring

    :param dataset: dataset object
    :param model_id: name or path of the model
    :param table_name: name of the db table in which the results are stored
    :param prompt_col_name: name of the dataset column containing the prompts
    :param result_col_name: name of the db table column in which to write the results
    :param batch_size: number of rows per batch, defaults to 20
    """
    llm = LLM(model=model_id, max_model_len=2048, dtype=torch.float16)  # Create an LLM.
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        skip_special_tokens=True,
        top_k=50,
        n=n_sequences,
    )
    start_idx = 0

    with tqdm(total=len(dataset)) as pbar:
        pbar.set_description("Generating DPO Samples")

        while start_idx < len(dataset):
            prompts = dataset[prompt_col_name][start_idx : start_idx + batch_size]
            orig_texts = dataset["content"][start_idx : start_idx + batch_size]
            try:
                outputs = generate_vllm(prompts, llm, sampling_params, use_tqdm=False)

                # get results and create IDs
                texts = [
                    o.text
                    for output in outputs
                    for o in output.outputs
                    if o.finish_reason == "stop"
                ]
                prompt_ids = [
                    md5(output.prompt.encode("utf-8")).hexdigest()
                    for output in outputs
                    for o in output.outputs
                    if o.finish_reason == "stop"
                ]
                prompts_formatted = [
                    output.prompt
                    for output in outputs
                    for o in output.outputs
                    if o.finish_reason == "stop"
                ]
                text_ids = [md5(t.encode("utf-8")).hexdigest() for t in texts]
                orig_texts_per_gen = [
                    t for prompt in prompts_formatted for t in orig_texts if t in prompt
                ]

                # store the result in a database
                result = {
                    "id": text_ids,
                    "prompt_id": prompt_ids,
                    "prompt": prompts_formatted,
                    "text_orig": orig_texts_per_gen,
                    result_col_name: texts,
                }

                df = pd.DataFrame(result)
                ingest_pandas(df, table_name)
            except (AssertionError, ValueError) as e:
                logger.debug(e)
                logger.debug(prompts)
            start_idx += batch_size
            pbar.update(batch_size)
    return


def process_vllm_outputs(outputs, orig_texts: list[str]):
    # get results and create IDs
    texts = [
        o.text
        for output in outputs
        for o in output.outputs
        if o.finish_reason == "stop"
    ]
    prompt_ids = [
        md5(output.prompt.encode("utf-8")).hexdigest()
        for output in outputs
        for o in output.outputs
        if o.finish_reason == "stop"
    ]
    prompts_formatted = [
        output.prompt
        for output in outputs
        for o in output.outputs
        if o.finish_reason == "stop"
    ]
    text_ids = [md5(t.encode("utf-8")).hexdigest() for t in texts]
    orig_texts_per_gen = [
        t for prompt in prompts_formatted for t in orig_texts if t in prompt
    ]

    # store the result in a database
    result = {
        "id": text_ids,
        "prompt_id": prompt_ids,
        "prompt": prompts_formatted,
        "text_orig": orig_texts_per_gen,
        "generated": texts,
    }

    df = pd.DataFrame(result)
    return df


def generate_raw_dpo_dataset(model_name: str, target_size: int = 1000):
    # todo docstring
    dataset = build_standard_german_dataset(model_name, target_size=target_size)
    prompts = create_prompts(dataset["content"])  # todo constants
    dataset = dataset.add_column(name="prompts", column=prompts)  # todo constants

    # run_vllm_batch_generation(
    #     dataset=dataset,
    #     model_id=model_name,
    #     result_table_name="dpo_raw_generations",
    #     ds_prompt_col_name="prompts",
    #     batch_size=10,
    #     max_model_len=2048,
    #     process_output=process_vllm_outputs,
    #     output_fn_kwargs={"orig_texts": dataset["content"]}
    # )
    run_vllm_generation(  # todo constants
        dataset=dataset,
        model_id=model_name,
        table_name="dpo_raw_generations",
        prompt_col_name="prompts",
        result_col_name="generated",
    )
    return


def score_dpo_generations(classification_model: str, batch_size: int = 100):
    conn = get_connector()
    sql = f"""
        SELECT * FROM dpo_raw_generations
        WHERE id NOT IN (SELECT id FROM dpo_scored_generations);
    """
    tokenizer = AutoTokenizer.from_pretrained(classification_model)
    model = AutoModelForSequenceClassification.from_pretrained(classification_model)
    dataset = Dataset.from_sql(sql, con=conn)

    start_idx = 0
    with tqdm(total=len(dataset)) as pbar:
        pbar.set_description("Scoring DPO generations")
        while start_idx < len(dataset):
            texts = dataset["generated"][start_idx : start_idx + batch_size]
            orig_texts = dataset["text_orig"][start_idx : start_idx + batch_size]
            df = calculate_metrics(
                model=model,
                tokenizer=tokenizer,
                texts=texts,
                original_texts=orig_texts,
                disable_tqdm=True,
            )
            df["id"] = dataset["id"][start_idx : start_idx + batch_size]
            df["prompt_id"] = dataset["prompt_id"][start_idx : start_idx + batch_size]
            df["prompt"] = dataset["prompt"][start_idx : start_idx + batch_size]
            ingest_pandas(df, "dpo_scored_generations")
            start_idx += batch_size
            pbar.update(batch_size)
    return


def sort_dpo_generations():
    conn = get_connector()
    sql = f"""
        SELECT * FROM dpo_scored_generations
    """
    df = pd.read_sql(sql, conn)
    dfs = []
    prompt_ids = set(list(df.prompt_id))
    for prompt_id in tqdm(prompt_ids, desc="Sorting into chosen and rejected"):
        prompt_df = df[df.prompt_id == prompt_id]
        # sort into chosen and rejected
        chosen_df = prompt_df[
            (prompt_df.predicted_class == 1)
            & (prompt_df.diff_logits > 9)
            & (prompt_df.share_newlines > 2)
            & (prompt_df.wiener_sachtextformel < 10)
            & (prompt_df.rouge2 < 0.65)
            & (prompt_df.shared_token_len > 0.5)
        ]
        rejected_df = prompt_df[~prompt_df.id.isin(chosen_df.id)]
        if len(chosen_df) > 0 and len(rejected_df) > 0:
            dpo_df = chosen_df.merge(
                rejected_df, how="cross", suffixes=["_chosen", "_rejected"]
            )[["prompt_chosen", "text_gen_chosen", "text_gen_rejected"]]
            dfs.append(dpo_df)
        # look for patterns to corrupt - later

    res_df = pd.concat(dfs)
    res_df = res_df.rename(
        columns={
            "prompt_chosen": "prompt",
            "text_gen_chosen": "chosen",
            "text_gen_rejected": "rejected",
        }
    )  # todo constants
    ingest_pandas(res_df, "dpo_paired_data", if_exists="replace")  # todo constants
    return


def create_hf_dpo_dataset():
    # todo docstring
    conn = get_connector()
    sql = f"""
        SELECT * FROM dpo_paired_data
    """
    dataset = Dataset.from_sql(sql, con=conn)
    # todo add to env vars
    push_to_hf_hub(dataset, "nsaef/German_leichte_sprache_dpo")


if __name__ == "__main__":
    args = parse_args()
    generate_raw_dpo_dataset(args.model_name, target_size=int(args.dataset_target_size))
    score_dpo_generations(args.classification_model)
    sort_dpo_generations()
    create_hf_dpo_dataset()
