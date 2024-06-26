from typing import Callable

from datasets import Dataset
import pandas as pd
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
     vLLM outputs (`list[RequestOutput]`), the batc as a mandatory parameter, optional keyword
     arguments (`output_fn_kwargs`) and must return a pandas DataFrame. The DataFrame
     will then be inserted into the DB.

     Example function:
     ```
     def process_output(outputs, start_idx, end_idx, ids: list) -> pd.DataFrame():
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
    :param sampling_params: sampling parameters
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
                result = process_output(
                    outputs, start_idx, start_idx + batch_size, **output_fn_kwargs
                )
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
