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
    DPO_PAIRS,
    DPO_RAW,
    DPO_SCORED,
    SRC_COLUMN,
    URL_COLUMN,
    CONTENT_COLUMN,
    PROMPTS_COLUMN,
    TEXT_COLUMN,
    ID_COLUMN,
    PROMPT_COLUMN,
    PROMPT_ID_COLUMN,
    GENERATED_COLUMN,
    TEXT_ORIG_COLUMN,
    TEXT_COLUMN,
    CHOSEN,
    REJECTED,
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
    """Get a list of domains that host desirable/usable content.

    :return: list of domain names
    """
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


def get_wiki_dataset() -> Dataset:
    """Load a dataset containing selected high-quality Wikipedia articles.
    Split the articles into sections separated by the article (sub-) headlines
    to create texts with a usable length. Create a new dataset with the columns
    `content`, `source`, `url` and the split articles as row.

    :return: split wikipedia dataset
    """
    wiki_dataset = load_dataset("PatrickHaller/wikitext-18-de", split="train").shuffle()
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
            CONTENT_COLUMN: articles,
            SRC_COLUMN: ["wiki" for _ in range(len(articles))],
            URL_COLUMN: urls,
        }
    )
    return wiki_ds


def get_web_dataset(target_size: int = 3000) -> Dataset:
    """Load the German split of a large web dataset and iterate through
    it until the specified number of articles from allow-listed websites
    has been collected. The allowlist contains regional and national news
    websites.

    :param target_size: amount of articles to collect, defaults to 3000
    :return: Dataset of specified size with columns `content`, `source`, `url`
    """
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
    web_ds = Dataset.from_list(articles).rename_column("text", CONTENT_COLUMN)
    return web_ds


def get_news_dataset() -> Dataset:
    """Load a news dataset containign German tagesschau articles.
    Combine headline and article text in new column `content`.

    :return: Dataset with columns `content`, `source`, `url`
    """
    news_dataset = (
        load_dataset("bjoernp/tagesschau-010124-020524", split="train")
        .map(
            lambda x: {
                CONTENT_COLUMN: f"{x['headline']}\n{x['article']}",
                SRC_COLUMN: "tagesschau",
            }
        )
        .rename_column("link", URL_COLUMN)
        .shuffle()
    )
    return news_dataset


def build_standard_german_dataset(
    tokenizer_path: str, target_size: int, model_max_length: int = 2048
) -> Dataset:
    """Build a dataset of standard German texts from sources that
    publish texts similar to the training data (mainly news, wiki).
    To create the dataset, first, set an initial target size. The dataset
    will be constructed from all avalable sources equally to reach this
    target size. It is then filtered to exclude texts longer than 2048
    tokens (semi-arbitraty threshold due to compute limitations), to avoid a
    large share of generations that stop in the middle of the output due to
    length constraints. This filters about 30% of the dataset.
    # todo: add max tokens to CLI

    :param tokenizer_path: path to the tokenizer of the intended generation model
    :param target_size: initial dataset size (before length filter)
    :param model_max_length: max number of tokens including the prompt, defaults to 2048
    :return: Dataset with texts to use for artifical DPO data generation
    """
    size_per_dataset = round(target_size / 3)

    wiki_ds = get_wiki_dataset()
    web_ds = get_web_dataset(target_size=size_per_dataset)
    news_ds = get_news_dataset()
    valid_columns = [CONTENT_COLUMN, SRC_COLUMN, URL_COLUMN]
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
        texts=dataset[CONTENT_COLUMN],
        prompt_len=tokens_prompt,
        model_max_len=model_max_length,
    )
    dataset = dataset.add_column("usable", text_len)
    dataset = dataset.filter(lambda x: x["usable"] is True)
    return dataset


def calculate_usable_text_length(
    tokenizer: AutoTokenizer, texts: list[str], prompt_len: int, model_max_len: int
) -> list[bool]:
    """Check if the texts in a list are usable or too long for a generative model.
    They are considered too long if it's unlikely that the model will be able to fit
    the prompt, the input text and the response into its maximum length.

    :param tokenizer:tokenizer of the generative model
    :param texts: list of input textz
    :param prompt_len: length of the prompt in tokens (tokenized with the correct tokenizer)
    :param model_max_len: max length of the generative model
    :return: list of bools per text (True => text can be used, False => text is too long)
    """
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

    :param dataset: dataset object
    :param model_id: name or path of the model
    :param table_name: name of the db table in which the results are stored
    :param prompt_col_name: name of the dataset column containing the prompts
    :param result_col_name: name of the db table column in which to write the results
    :param batch_size: number of rows per batch, defaults to 10
    :params n_sequences: number of return sequences per prompt, defaults to 5
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
            orig_texts = dataset[CONTENT_COLUMN][start_idx : start_idx + batch_size]
            try:
                outputs = generate_vllm(prompts, llm, sampling_params, use_tqdm=False)
                df = process_vllm_outputs(outputs, orig_texts=orig_texts)
                ingest_pandas(df, table_name)
            except (AssertionError, ValueError) as e:
                logger.debug(e)
                logger.debug(prompts)
            start_idx += batch_size
            pbar.update(batch_size)
    return


def process_vllm_outputs(outputs, orig_texts: list[str]) -> pd.DataFrame:
    """Process the vLLM outputs and reformat them into a DataFrame that
    can be inserted into the project DB.

    :param outputs: list of vLLM RequestOutput objects
    :param orig_texts: list of original texts_
    :return: dataframe containing the generated texts with metadata
    """
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
        ID_COLUMN: text_ids,
        PROMPT_ID_COLUMN: prompt_ids,
        PROMPT_COLUMN: prompts_formatted,
        TEXT_ORIG_COLUMN: orig_texts_per_gen,
        GENERATED_COLUMN: texts,
    }

    df = pd.DataFrame(result)
    return df


def generate_raw_dpo_dataset(model_name: str, target_size: int = 1000):
    """Generate the basis for a DPO dataset by creating several variants
    in Leichte Sprache for a number of prompts in standard German.

    With the default settings, a standard German dataset with 1000 rows is
    constructed from multiple sources. This dataset is then filtered to
    exclude texts that likely cannot be fit into the model with a complete
    output (shrinks to ~70%). For each of the remaining rows, five texts are
    generated, resulting in 3000-3500 usable generations, as only generations
    that finish due to a stop token being generated are used.

    :param model_name: generative model that can produce Leichte Sprache
    :param target_size: target size for the standard German dataset, defaults to 1000
    """
    dataset = build_standard_german_dataset(model_name, target_size=target_size)
    prompts = create_prompts(dataset[CONTENT_COLUMN])
    dataset = dataset.add_column(name=PROMPTS_COLUMN, column=prompts)

    # note: this refactored function would be preferable, but isn't ready yet
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
    run_vllm_generation(
        dataset=dataset,
        model_id=model_name,
        table_name=DPO_RAW,
        prompt_col_name=PROMPTS_COLUMN,
        result_col_name=GENERATED_COLUMN,
    )
    return


def score_dpo_generations(classification_model: str, batch_size: int = 100):
    """
    Load all rows of generated data from the database that haven't been scored
    yet. Calculate a number of metrics on them, such as rouge2, Flesch Reading Ease,
    and the logits of the specifically-trained Leichte Sprache classifier.
    Store the result in a new table of the project DB.
    """
    conn = get_connector()
    sql = f"""
        SELECT * FROM {DPO_RAW}
        WHERE {ID_COLUMN} NOT IN (SELECT {ID_COLUMN} FROM {DPO_SCORED});
    """
    tokenizer = AutoTokenizer.from_pretrained(classification_model)
    model = AutoModelForSequenceClassification.from_pretrained(classification_model)
    dataset = Dataset.from_sql(sql, con=conn)

    start_idx = 0
    with tqdm(total=len(dataset)) as pbar:
        pbar.set_description("Scoring DPO generations")
        while start_idx < len(dataset):
            texts = dataset[GENERATED_COLUMN][start_idx : start_idx + batch_size]
            orig_texts = dataset[TEXT_ORIG_COLUMN][start_idx : start_idx + batch_size]
            df = calculate_metrics(
                model=model,
                tokenizer=tokenizer,
                texts=texts,
                original_texts=orig_texts,
                disable_tqdm=True,
            )
            df[ID_COLUMN] = dataset[ID_COLUMN][start_idx : start_idx + batch_size]
            df[PROMPT_ID_COLUMN] = dataset[PROMPT_ID_COLUMN][
                start_idx : start_idx + batch_size
            ]
            df[PROMPT_COLUMN] = dataset[PROMPT_COLUMN][
                start_idx : start_idx + batch_size
            ]
            ingest_pandas(df, DPO_SCORED)
            start_idx += batch_size
            pbar.update(batch_size)
    return


def sort_dpo_generations():
    """
    Sort the DPO generations into pairs of chosen and rejected. To do this,
    load all scored generations from the DB and group them by prompt. Depending
    on the scores of the generations, sort them into chosen/rejected and pair
    all instances of both groups with each other.
    Drop the existing DPO paired data table and replace it with the new set of pairs.
    """
    conn = get_connector()
    sql = f"""
        SELECT * FROM {DPO_SCORED}
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
                rejected_df, how="cross", suffixes=[f"_{CHOSEN}", f"_{REJECTED}"]
            )[[f"prompt_{CHOSEN}", f"text_gen_{CHOSEN}", f"text_gen_{REJECTED}"]]
            dfs.append(dpo_df)
        # look for patterns to corrupt - later

    res_df = pd.concat(dfs)
    res_df = res_df.rename(
        columns={
            f"{PROMPT_COLUMN}_{CHOSEN}": PROMPT_COLUMN,
            f"text_gen_{CHOSEN}": CHOSEN,
            f"text_gen_{REJECTED}": REJECTED,
        }
    )
    ingest_pandas(res_df, DPO_PAIRS, if_exists="replace")
    return


def create_hf_dpo_dataset():
    """
    Load all DPO pairs from the database and push them to a dataset
    on the HuggingFace hub.
    """
    conn = get_connector()
    sql = f"""
        SELECT * FROM {DPO_PAIRS}
    """
    dataset = Dataset.from_sql(sql, con=conn)
    # todo add to env vars
    push_to_hf_hub(dataset, "nsaef/German_leichte_sprache_dpo")
    return


if __name__ == "__main__":
    args = parse_args()
    generate_raw_dpo_dataset(args.model_name, target_size=int(args.dataset_target_size))
    score_dpo_generations(args.classification_model)
    sort_dpo_generations()
    create_hf_dpo_dataset()
