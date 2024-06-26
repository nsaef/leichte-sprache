from typing import Callable

from datasets import Dataset
import pandas as pd
import tiktoken
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams, RequestOutput

from leichte_sprache.utils.db_utils import ingest_pandas
from leichte_sprache.utils.utils import get_logger

logger = get_logger()


def run_vllm_batch_generation(
    dataset: Dataset,
    model_id: str,
    result_table_name: str,
    ds_prompt_col_name: str,
    process_output: Callable,
    output_fn_kwargs: dict = None,
    batch_size: int = 20,
    max_model_len: int = 1024,
    sampling_params: SamplingParams = None,
    tqdm_desc="Running vLLM generation",
):
    """Generate output from prompts in a dataset using VLLM. The prompts are batched
     according the the `batch_size` parameter. The outputs of each batch are stored
     in a database. In order to store the outputs in a database in the correct format,
     it is necessary to provide a function to process the outputs. This function takes
     vLLM outputs (`list[RequestOutput]`) as a mandatory parameter, optional keyword
     arguments (`output_fn_kwargs`) and must return a pandas DataFrame. The DataFrame
     will then be inserted into the DB.

     Example function:
     ```
     def process_output(outputs, ids: list) -> pd.DataFrame():
        texts = [o.text for output in outputs for o in output.outputs if o.finish_reason == "stop"]
        res = {"text": texts, "id": ids}
        df = pd.DataFrame(res)
        return df
    ```

    Default sampling parameters:
    ```
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        skip_special_tokens=True,
        top_k=50,
        n=1,
    )
    ```

    :param dataset: dataset object
    :param model_id: name or path of the model
    :param result_table_name: name of the db table in which the results are stored
    :param ds_prompt_col_name: name of the dataset column containing the prompts
    :param process_output: function processing the vLLM output into a pandas DataFrame
    :param output_function_kwargs: keyword-arguments to pass to `process_output`
    :param batch_size: number of rows per batch, defaults to 20
    :param max_model_len: maximum model input length, defaults to 1024
    :param sampling_params:
    """
    llm = LLM(model=model_id, max_model_len=max_model_len, dtype=torch.float16)
    if not sampling_params:
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.9,
            skip_special_tokens=True,
            top_k=50,
            n=1,
        )
    if not output_fn_kwargs:
        output_fn_kwargs = {}
    start_idx = 0

    with tqdm(total=len(dataset)) as pbar:
        pbar.set_description(tqdm_desc)

        while start_idx < len(dataset):
            prompts = dataset[ds_prompt_col_name][start_idx : start_idx + batch_size]
            try:
                outputs = generate_vllm(prompts, llm, sampling_params, use_tqdm=False)
                result = process_output(outputs, **output_fn_kwargs)
                df = pd.DataFrame(result)
                ingest_pandas(df, result_table_name)
            except AssertionError as e:
                logger.debug(e)
                logger.debug(prompts)
            start_idx += batch_size
            pbar.update(batch_size)
    return


def generate_vllm(
    messages: list,
    llm: LLM,
    sampling_params: SamplingParams,
    use_tqdm: bool = True,
) -> list[RequestOutput]:
    """Generate with a transformers-compatible model using VLLM. Messages are
    converted to plain text via the chat template and then passed to the model.

    :param messages: list of prompts in the chat messages format
    :param llm: VLLM LLM instance
    :param sampling_params: sampling parameters
    :param use_tqdm: show a tqdm progress bar. Default: True
    :return: list of VLLM RequestOutput objects
    """
    tokenizer = llm.llm_engine.tokenizer.tokenizer
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    sampling_params.stop_token_ids = terminators
    sampling_params.max_tokens = (llm.llm_engine.model_config.max_model_len,)
    prompts = [
        tokenizer.apply_chat_template(
            message, add_generation_prompt=True, tokenize=False
        )
        for message in messages
    ]
    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=use_tqdm,
    )
    return outputs


def count_tokens_openai(model: str, texts: list[str]) -> list[int]:
    """Count tokens for a batch of texts using the OpenAI library tiktoken.

    :param model: OpenAI modelname
    :param texts: list of texts
    :return: list with the number of tokens of each text
    """
    encoding = tiktoken.encoding_for_model(model)
    tokenized = encoding.encode_batch(texts)
    n_tokens = [len(t) for t in tokenized]
    return n_tokens


def count_tokens_from_messages_openai(model: str, messages: list[dict]):
    """Return the number of tokens used by a list of messages.

    :param model: OpenAI modelname
    :param messages: list of dicst in the chat message format
    :return: number of tokens in the messages
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print(
            "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
        )
        return count_tokens_from_messages_openai(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print(
            "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
        )
        return count_tokens_from_messages_openai(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
